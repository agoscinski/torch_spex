import copy
from torch_nl import compute_neighborlist

import numpy as np
import torch
import ase
from ase.neighborlist import primitive_neighbor_list
import equistore.operations
from equistore.torch import TensorMap, Labels, TensorBlock

from .angular_basis import AngularBasis
from .radial_basis import RadialBasis

from typing import Dict, List, Union

class SphericalExpansion(torch.nn.Module):
    """
    The spherical expansion coefficients summed over all neighbours.

    .. math::

         \sum_j c^{l}_{Aija_ia_j, m, n} = c^{l}_{Aia_ia_j, m, n}
         --reorder--> c^{a_il}_{Ai, m, a_jn}

    where:
    - **A**: index atomic structure,
    - **i**: index of central atom,
    - **j**: index of neighbor atom,
    - **a_i**: species of central atom,
    - **a_j**: species of neighbor atom or pseudo species,
    - **n**: radial channel corresponding to n'th radial basis function,
    - **l**: degree of spherical harmonics,
    - **m**: order of spherical harmonics

    The indices of the coefficients are written to show the storage in an
    equistore.TensorMap object

    .. math::

         c^{keys}_{samples, components, properties}

    :param hypers:
        - **cutoff radius**: cutoff for the neighborlist
        - **radial basis**: smooth basis optimizing Rayleight quotients [lle]_
          - **E_max** energy cutoff for the eigenvalues of the eigenstates
        - **alchemical**: number of pseudo species to reduce the species channels to

    .. [lle]
        Bigi, Filippo, et al. "A smooth basis for atomistic machine learning."
        The Journal of Chemical Physics 157.23 (2022): 234101.
        https://doi.org/10.1063/5.0124363

    >>> import numpy as np
    >>> from ase.build import molecule
    >>> from torch_spex.structures import ase_atoms_to_tensordict
    >>> from torch_spex.spherical_expansions import SphericalExpansion
    >>> hypers = {
    ...     "cutoff radius": 3,
    ...     "radial basis": {
    ...         "E_max": 20
    ...     },
    ...     "alchemical": 1,
    ... }
    >>> h2o = molecule("H2O")
    >>> spherical_expansion = SphericalExpansion(hypers, [1,8], device="cpu")
    >>> atomic_structures = ase_atoms_to_tensordict([h2o])
    >>> spherical_expansion.forward(atomic_structures)
    TensorMap with 2 blocks
    keys: ['a_i' 'lam' 'sigma']
             1     0      1
             8     0      1

    """

    def __init__(self, hypers: Dict, all_species: List[int], device: str ="cpu") -> None:
        super().__init__()

        self.hypers = hypers
        self.all_species = torch.tensor(all_species, dtype=torch.int32)
        self.vector_expansion_calculator = VectorExpansion(hypers, device=device)
        # we have to assign it outside of if because torchscript does not find it otherwise
        self.all_species_labels = Labels(
            names = ["species_neighbor"],
            values = torch.tensor(self.all_species[:, None], dtype=torch.int32)
        )
        if "alchemical" in self.hypers:
            self.is_alchemical = True
            self.n_pseudo_species = self.hypers["alchemical"]
            self.combination_matrix = torch.nn.Linear(self.all_species.shape[0], self.n_pseudo_species, bias=False, dtype=torch.get_default_dtype())
        else:
            self.is_alchemical = False
            self.n_pseudo_species = 0
            self.combination_matrix = torch.nn.Linear(self.all_species.shape[0], self.n_pseudo_species, bias=False, dtype=torch.get_default_dtype())
        self.default_type = torch.get_default_dtype()

    def forward(self, structures: Dict[str, torch.Tensor]):

        expanded_vectors = self.vector_expansion_calculator(structures)
        #samples_metadata = expanded_vectors.block(l=0)
        # error message
        #RuntimeError: block() is missing value for argument 'selection'. Declaration: block(__torch__.torch.classes.equistore.TensorMap _0, Any selection) -> __torch__.torch.classes.equistore.TensorBlock _0

        l0_labels = Labels(names=["l"], values=torch.tensor([(0,)]))
        samples_metadata = expanded_vectors.block(l0_labels).samples
        # does not work because it is a ScriptObject

        # TMP hack needs to be changed before merge
        #samples_metadata = expanded_vectors.block(0)
        #samples_metadata_structure: torch.Tensor = samples_metadata["structure"].values
        s_metadata = samples_metadata.values[:, samples_metadata.names.index("structure")].flatten()
        i_metadata = samples_metadata.values[:, samples_metadata.names.index("center")].flatten()
        ai_metadata = samples_metadata.values[:, samples_metadata.names.index("species_center")].flatten()

        # does not work with torchscript
        #s_metadata = samples_metadata["structure"].values.flatten()
        #i_metadata = samples_metadata["center"].values.flatten()
        #ai_metadata = samples_metadata["species_center"].values.flatten()

        n_species = len(self.all_species)
        species_to_index = {int(atomic_number) : i_species for i_species, atomic_number in enumerate(self.all_species)}

        s_i_metadata = torch.stack([s_metadata, i_metadata], dim=-1)
        unique_s_i_indices, s_i_unique_to_metadata, s_i_metadata_to_unique = torch_unique(s_i_metadata)

        l_max = self.vector_expansion_calculator.l_max
        n_centers = len(unique_s_i_indices)

        densities = []
        if self.is_alchemical:

            l0_labels = Labels(names=["l"], values=torch.tensor([(0,)]))
            one_hot_aj = torch.tensor(
                    one_hot(samples_metadata, self.all_species_labels),
                    dtype = self.default_type,
                    device = expanded_vectors.block(l0_labels).values.device
                    )
            pseudo_species_weights = self.combination_matrix(one_hot_aj)

            density_indices = torch.LongTensor(s_i_metadata_to_unique)
            for l in range(l_max+1):
                l_labels = Labels(names=["l"], values=torch.tensor([(l,)]))
                expanded_vectors_l = expanded_vectors.block(l_labels ).values
                expanded_vectors_l_pseudo = torch.einsum(
                        "abc, ad -> abcd", expanded_vectors_l, pseudo_species_weights
                        )
                densities_l = torch.zeros(
                        (n_centers, expanded_vectors_l.shape[1], expanded_vectors_l.shape[2], self.n_pseudo_species),
                        dtype = expanded_vectors_l.dtype,
                        device = expanded_vectors_l.device
                        )
                densities_l.index_add_(dim=0, index=density_indices.to(expanded_vectors_l.device), source=expanded_vectors_l_pseudo)
                densities_l = densities_l.reshape((n_centers, 2*l+1, -1, self.n_pseudo_species)).swapaxes(2, 3).reshape((n_centers, 2*l+1, -1))
                densities.append(densities_l)
            species = -torch.arange(self.n_pseudo_species)
        else:
            aj_metadata = samples_metadata.values[:, samples_metadata.names.index("species_neighbor")]
            aj_shifts = torch.tensor([species_to_index[int(aj_index)] for aj_index in aj_metadata])
            density_indices = torch.LongTensor(s_i_metadata_to_unique*n_species+aj_shifts)

            for l in range(l_max+1):
                l_labels = Labels(names=["l"], values=torch.tensor([(l,)]))
                expanded_vectors_l = expanded_vectors.block(l_labels).values
                densities_l = torch.zeros(
                        (n_centers*n_species, expanded_vectors_l.shape[1], expanded_vectors_l.shape[2]),
                        dtype = expanded_vectors_l.dtype,
                        device = expanded_vectors_l.device
                        )
                densities_l.index_add_(dim=0, index=density_indices.to(expanded_vectors_l.device), source=expanded_vectors_l)
                densities_l = densities_l.reshape((n_centers, n_species, 2*l+1, -1)).swapaxes(1, 2).reshape((n_centers, 2*l+1, -1))
                densities.append(densities_l)
            species = torch.tensor(self.all_species, dtype=torch.int32)

        # constructs the TensorMap object
        ai_new_indices = torch.tensor(ai_metadata[s_i_unique_to_metadata])
        labels: List[List[int]] = []
        blocks: List[TensorBlock] = []
        for l in range(l_max+1):
            densities_l = densities[l]
            l_labels = Labels(names=["l"], values=torch.tensor([(l,)]))
            vectors_l_block = expanded_vectors.block(l_labels )
            vectors_l_block_components = vectors_l_block.components
            # does not work form some reason I get 
            #vectors_l_block_properties_view = vectors_l_block.properties["n"]
            #vectors_l_block_properties_view = vectors_l_block.properties.names.index("n")
            #vectors_l_block_properties_view = vectors_l_block.properties.values[:, vectors_l_block.properties.names.index("n")].flatten()
            #breakpoint()
            #assert isinstance(vectors_l_block_properties_view, Labels)
            #TypeError: isinstance() arg 2 must be a type, a tuple of types, or a union

            #vectors_l_block_n = vectors_l_block_properties_view.values.flatten()
            vectors_l_block_n = vectors_l_block.properties.values[:, vectors_l_block.properties.names.index("n")].flatten()

            samples_metadata.values[:, samples_metadata.names.index("species_center")]
            for a_i in self.all_species:
                where_ai = torch.LongTensor(torch.where(ai_new_indices == a_i)[0]).to(densities_l.device)
                densities_ai_l = torch.index_select(densities_l, 0, where_ai)
                a_i_int: int = int(a_i)
                l_int: int = int(l) # MAYBE NOT NEEDED
                labels.append([a_i_int, l, 1])
                blocks.append(
                        TensorBlock(
                            values = densities_ai_l,
                            samples = Labels(
                                names = ["structure", "center"],
                                values = unique_s_i_indices[where_ai.cpu()]
                                ),
                            components = vectors_l_block_components,
                            properties = Labels(
                                names = ["a1", "n1", "l1"],
                                values = torch.stack(
                                    [
                                        species.repeat(vectors_l_block_n.shape[0]),
                                        torch.tile(vectors_l_block_n, (species.shape[0],)),
                                        l*torch.ones((densities_ai_l.shape[2],), dtype=torch.int32)
                                        ],
                                    dim=1
                                    )
                                )
                            )
                        )

        spherical_expansion = TensorMap(
                keys = Labels(
                    names = ["a_i", "lam", "sigma"],
                    values = torch.tensor(labels, dtype=torch.int32)
                    ),
                blocks = blocks
                )

        return spherical_expansion


class VectorExpansion(torch.nn.Module):
    """
    The spherical expansion coefficients for each neighbour

    .. math::

        c^{l}_{Aija_ia_j,m,n}

    where:
    - **A**: index atomic structure,
    - **i**: index of central atom,
    - **j**: index of neighbor atom,
    - **a_i**: species of central atom,
    - **a_j**: species of neighbor aotm,
    - **n**: radial channel corresponding to n'th radial basis function,
    - **l**: degree of spherical harmonics,
    - **m**: order of spherical harmonics

    The indices of the coefficients are written to show the storage in an
    equistore.TensorMap object

    .. math::

         c^{keys}_{samples, components, properties}

    """

    def __init__(self, hypers: Dict[str, Union[float,int]], device: str ="cpu") -> None:
        super().__init__()

        self.hypers = hypers
        # radial basis needs to know cutoff so we pass it
        hypers_radial_basis = copy.deepcopy(hypers["radial basis"])
        hypers_radial_basis["r_cut"] = hypers["cutoff radius"]
        self.cutoff_radius = torch.tensor(hypers["cutoff radius"], dtype=torch.get_default_dtype()) # 
        #self.cutoff_radius = float(hypers["cutoff radius"]) # type : float
        self.radial_basis_calculator = RadialBasis(hypers_radial_basis, device=device)
        self.l_max = self.radial_basis_calculator.l_max
        self.spherical_harmonics_calculator = AngularBasis(self.l_max)

    def forward(self, structures: Dict[str, torch.Tensor]):

        cartesian_vectors = get_cartesian_vectors(structures, self.cutoff_radius)

        bare_cartesian_vectors = cartesian_vectors.values.squeeze(dim=-1)

        r = torch.sqrt(
                (bare_cartesian_vectors**2)
                .sum(dim=-1)
                )
        radial_basis = self.radial_basis_calculator(r)

        spherical_harmonics = self.spherical_harmonics_calculator(bare_cartesian_vectors)

        # Use broadcasting semantics to get the products in equistore shape
        # torchscript cannot inference type of list
        l = 0
        vector_expansion_l = radial_basis[0].unsqueeze(dim = 1) * spherical_harmonics[0].unsqueeze(dim = 2)
        n_max_l = vector_expansion_l.shape[2]
        block = TensorBlock(
                values = vector_expansion_l,
                samples = cartesian_vectors.samples,
                components = [Labels(
                    names = ["m"],
                    values = torch.arange(-l, l+1, dtype=torch.int32).reshape(2*l+1, 1)
                    )],
                properties = Labels(
                    names = ["n"],
                    values = torch.arange(0, n_max_l, dtype=torch.int32).reshape(n_max_l, 1)
                    )
                )
        vector_expansion_blocks = [block]
        for l, (radial_basis_l, spherical_harmonics_l) in enumerate(zip(radial_basis, spherical_harmonics)):
            if l == 0:
                continue
            vector_expansion_l = radial_basis_l.unsqueeze(dim = 1) * spherical_harmonics_l.unsqueeze(dim = 2)
            n_max_l = vector_expansion_l.shape[2]
            block = TensorBlock(
                    values = vector_expansion_l,
                    samples = cartesian_vectors.samples,
                    components = [Labels(
                        names = ["m"],
                        values = torch.arange(-l, l+1, dtype=torch.int32).reshape(2*l+1, 1)
                        )],
                    properties = Labels(
                        names = ["n"],
                        values = torch.arange(0, n_max_l, dtype=torch.int32).reshape(n_max_l, 1)
                        )
                    )
            vector_expansion_blocks.append(block)


        l_max = len(vector_expansion_blocks) - 1
        vector_expansion_tmap = TensorMap(
                keys = Labels(
                    names = ["l"],
                    values = torch.arange(0, l_max+1, dtype=torch.int32).reshape(l_max+1, 1),
                    ),
                blocks = vector_expansion_blocks
                )

        return vector_expansion_tmap


def get_cartesian_vectors(structures: Dict[str, torch.Tensor], cutoff_radius: torch.Tensor):

    labels = []
    vectors = []

    for structure_index in range(int(structures["n_structures"])):

        where_selected_structure = torch.where(structures["structure_indices"] == structure_index)[0]

        centers, neighbors, unit_cell_shift_vectors = get_neighbor_list(
                structures["positions"][where_selected_structure],
                structures["pbcs"][structure_index],
                structures["cells"][structure_index],
                cutoff_radius)

        positions = structures["positions"][torch.LongTensor(where_selected_structure)]
        cell = structures["cells"][structure_index]
        species = structures["atomic_species"][structure_index]

        structure_vectors = positions[neighbors] - positions[centers] + (unit_cell_shift_vectors @ cell).to(positions.device)  # Warning: it works but in a weird way when there is no cell
        vectors.append(structure_vectors)
        labels.append(
                torch.stack([
                    torch.tensor([structure_index]*len(centers), dtype=torch.int64),
                    centers,
                    neighbors,
                    species[centers],
                    species[neighbors],
                    unit_cell_shift_vectors[:, 0],
                    unit_cell_shift_vectors[:, 1],
                    unit_cell_shift_vectors[:, 2]
                    ], dim=-1))

    vectors = torch.cat(vectors, dim=0)
    labels = torch.concatenate(labels, dim=0)

    block = TensorBlock(
            values = vectors.unsqueeze(dim=-1),
            samples = Labels(
                names = ["structure", "center", "neighbor", "species_center", "species_neighbor", "cell_x", "cell_y", "cell_z"],
                values = torch.tensor(labels, dtype=torch.int32)
                ),
            components = [
                Labels(
                    names = ["cartesian_dimension"],
                    values = torch.tensor([-1, 0, 1], dtype=torch.int32).reshape((-1, 1))
                    )
                ],
            properties = Labels.single()
            )

    return block


def get_neighbor_list(positions, pbc, cell, cutoff_radius):


    #mapping, batch_mapping, shifts_idx = compute_neighborlist(cutoff_radius, positions, cell, pbc, torch.zeros(len(positions), dtype=torch.int64), self_interaction=True)
    mapping, _, unit_cell_shift_vectors = compute_neighborlist(
            cutoff_radius, positions, cell, pbc, torch.zeros(len(positions), dtype=torch.int64), self_interaction=True
            )
    centers = mapping[0]
    neighbors = mapping[1]

    #centers, neighbors, unit_cell_shift_vectors = ase.neighborlist.primitive_neighbor_list( quantities="ijS", pbc=pbc.detach().numpy(), cell=cell.detach().numpy(), positions=positions.detach().numpy(), cutoff=cutoff_radius, self_interaction=True, use_scaled_positions=False)
    #centers, neighbors, unit_cell_shift_vectors = ase.neighborlist.primitive_neighbor_list(
    #    quantities="ijS",
    #    pbc=pbc.detach().numpy(),
    #    cell=cell.detach().numpy(),
    #    positions=positions.detach().numpy(),
    #    cutoff=cutoff_radius,
    #    self_interaction=True,
    #    use_scaled_positions=False,
    #)

    pairs_to_throw = torch.logical_and(centers == neighbors, torch.all(unit_cell_shift_vectors == 0, dim=1))
    pairs_to_keep = torch.logical_not(pairs_to_throw)

    centers = centers[pairs_to_keep]
    neighbors = neighbors[pairs_to_keep]
    unit_cell_shift_vectors = unit_cell_shift_vectors[pairs_to_keep]

    #centers = torch.LongTensor(centers)
    #neighbors = torch.LongTensor(neighbors)
    #unit_cell_shift_vectors = torch.tensor(unit_cell_shift_vectors, dtype=torch.get_default_dtype())

    return centers, neighbors, unit_cell_shift_vectors

def torch_unique(x):
    """Unique elements of x and indices of those unique elements
    https://github.com/pytorch/pytorch/issues/36748#issuecomment-619514810

    imitategs return_index=True, return_inverse=True from numpy
    e.g.

    unique(tensor([
        [1, 2, 3],
        [1, 2, 4],
        [1, 2, 3],
        [1, 2, 5]
    ]), dim=0)
    => (tensor([[1, 2, 3],
                [1, 2, 4],
                [1, 2, 5]]),
        tensor([0, 1, 3]))
        tensor ... inverse
        """
    # hardcoded dim because torchscript complains
    unique, inverse = torch.unique(x, sorted=True, return_inverse=True, dim=0)
    perm = torch.arange(inverse.size(0), dtype=inverse.dtype, device=inverse.device)
    inverse, perm = inverse.flip([0]), perm.flip([0])
    perm = inverse.new_empty(unique.size(0)).scatter_(0, inverse, perm)
    return unique, perm, inverse.flip([0])

def one_hot(labels: Labels, dimension: Labels):

    if len(dimension.names) != 1:
        raise ValueError(
            "only one label dimension can be extracted as one-hot "
            "encoding. The `dimension` labels contains "
            f"{len(dimension.names)} names"
        )

    name = dimension.names[0]
    #possible_labels_view = dimension[name]
    #assert isinstance(possible_labels_view, Labels)
    #possible_labels = possible_labels_view.values.flatten()
    possible_labels = dimension.view([name]).values.flatten()
    if name not in labels.names:
        raise ValueError("the dimension provided was not found among the labels")
    original_labels = labels.view([name]).values

    indices = torch.where(
        original_labels.reshape(-1, 1) == possible_labels
    )[1]

    if indices.shape[0] != labels.values.shape[0]:
        raise ValueError(
            "some values not present in the dimension were found in the labels"
        )
    one_hot_array = torch.eye(possible_labels.shape[0])[indices]

    return one_hot_array
