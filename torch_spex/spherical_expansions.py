import copy

import numpy as np
import torch
import ase
from ase.neighborlist import primitive_neighbor_list
from .structures import get_cartesian_vectors
import equistore
from equistore import TensorMap, Labels, TensorBlock
import sphericart.torch

from .radial_basis import RadialBasis
from typing import Dict, List

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
    keys: a_i  lam  sigma
           1    0     1
           8    0     1

    """

    def __init__(self, hypers: Dict, all_species: List[int], device: str = None) -> None:
        super().__init__()

        self.hypers = hypers
        self.all_species = np.array(all_species, dtype=np.int32)  # convert potential list to np.array
        self.vector_expansion_calculator = VectorExpansion(hypers, device=device)

        if "alchemical" in self.hypers:
            self.is_alchemical = True
            self.all_species_labels = Labels(
                names = ["species_neighbor"],
                values = self.all_species[:, None]
            )
            self.n_pseudo_species = self.hypers["alchemical"]
            self.combination_matrix = torch.nn.Linear(self.all_species.shape[0], self.n_pseudo_species, bias=False)
        else:
            self.is_alchemical = False

    def forward(self, structures: Dict[str, torch.Tensor]):

        expanded_vectors = self.vector_expansion_calculator(structures)
        samples_metadata = expanded_vectors.block(l=0).samples

        s_metadata = samples_metadata["structure"]
        i_metadata = samples_metadata["center"]
        ai_metadata = samples_metadata["species_center"]

        n_species = len(self.all_species)
        species_to_index = {atomic_number : i_species for i_species, atomic_number in enumerate(self.all_species)}

        s_i_metadata = np.concatenate([s_metadata, i_metadata], axis=-1)
        unique_s_i_indices, s_i_unique_to_metadata, s_i_metadata_to_unique = np.unique(s_i_metadata, axis=0, return_index=True, return_inverse=True)

        l_max = self.vector_expansion_calculator.l_max
        n_centers = len(unique_s_i_indices)

        densities = []
        if self.is_alchemical:

            one_hot_aj = torch.tensor(
                equistore.one_hot(samples_metadata, self.all_species_labels),
                dtype = torch.get_default_dtype(),
                device = expanded_vectors.block(l=0).values.device
            )
            pseudo_species_weights = self.combination_matrix(one_hot_aj)

            density_indices = torch.LongTensor(s_i_metadata_to_unique)
            for l in range(l_max+1):
                expanded_vectors_l = expanded_vectors.block(l=l).values
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
            species = -np.arange(self.n_pseudo_species)
        else:
            aj_metadata = samples_metadata["species_neighbor"]
            for aj_index in aj_metadata:
                pass # print(aj_index)
            aj_shifts = np.array([species_to_index[aj_index] for aj_index in aj_metadata.values[:, 0]])
            density_indices = torch.LongTensor(s_i_metadata_to_unique*n_species+aj_shifts)

            for l in range(l_max+1):
                expanded_vectors_l = expanded_vectors.block(l=l).values
                densities_l = torch.zeros(
                    (n_centers*n_species, expanded_vectors_l.shape[1], expanded_vectors_l.shape[2]),
                    dtype = expanded_vectors_l.dtype,
                    device = expanded_vectors_l.device
                )
                densities_l.index_add_(dim=0, index=density_indices.to(expanded_vectors_l.device), source=expanded_vectors_l)
                densities_l = densities_l.reshape((n_centers, n_species, 2*l+1, -1)).swapaxes(1, 2).reshape((n_centers, 2*l+1, -1))
                densities.append(densities_l)
            species = self.all_species

        # constructs the TensorMap object
        ai_new_indices = torch.tensor(ai_metadata.values[s_i_unique_to_metadata])
        labels = []
        blocks = []
        for l in range(l_max+1):
            densities_l = densities[l]
            vectors_l_block = expanded_vectors.block(l=l)
            vectors_l_block_components = vectors_l_block.components
            vectors_l_block_n = torch.tensor(vectors_l_block.properties["n"].values[:, 0])
            for a_i in self.all_species:
                where_ai = torch.where(ai_new_indices == a_i)[0].to(densities_l.device)
                densities_ai_l = torch.index_select(densities_l, 0, where_ai)
                labels.append([a_i, l, 1])
                blocks.append(
                    TensorBlock(
                        values = densities_ai_l,
                        samples = Labels(
                            names = ["structure", "center"],
                            values = unique_s_i_indices[where_ai.cpu().numpy()]
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
                values = np.array(labels, dtype=np.int32)
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

    def __init__(self, hypers: Dict, device: str = None) -> None:
        super().__init__()

        self.hypers = hypers
        # radial basis needs to know cutoff so we pass it
        hypers_radial_basis = copy.deepcopy(hypers["radial basis"])
        hypers_radial_basis["r_cut"] = hypers["cutoff radius"]
        self.radial_basis_calculator = RadialBasis(hypers_radial_basis, device=device)
        self.l_max = self.radial_basis_calculator.l_max
        self.spherical_harmonics_calculator = sphericart.torch.SphericalHarmonics(self.l_max, normalized=True)
        self.spherical_harmonics_split_list = [(2*l+1) for l in range(self.l_max+1)]

    def forward(self, structures: Dict[str, torch.Tensor]):

        if "cartesian_vectors" in structures.keys():
            cartesian_vectors = structures["cartesian_vectors"]
        else:
            cutoff_radius = self.hypers["cutoff radius"]
            cartesian_vectors = get_cartesian_vectors(structures, cutoff_radius)

        bare_cartesian_vectors = cartesian_vectors.values.squeeze(dim=-1)

        r = torch.sqrt(
            (bare_cartesian_vectors**2)
            .sum(dim=-1)
        )
        radial_basis = self.radial_basis_calculator(r)

        spherical_harmonics = self.spherical_harmonics_calculator.compute(bare_cartesian_vectors)  # Get the spherical harmonics
        spherical_harmonics = torch.split(spherical_harmonics, self.spherical_harmonics_split_list, dim=1)  # Split them into l chunks

        # Use broadcasting semantics to get the products in equistore shape
        vector_expansion_blocks = []
        for l, (radial_basis_l, spherical_harmonics_l) in enumerate(zip(radial_basis, spherical_harmonics)):
            vector_expansion_l = radial_basis_l.unsqueeze(dim = 1) * spherical_harmonics_l.unsqueeze(dim = 2)
            n_max_l = vector_expansion_l.shape[2]
            vector_expansion_blocks.append(
                TensorBlock(
                    values = vector_expansion_l,
                    samples = cartesian_vectors.samples,
                    components = [Labels(
                        names = ("m",),
                        values = np.arange(-l, l+1, dtype=np.int32).reshape(2*l+1, 1)
                    )],
                    properties = Labels(
                        names = ("n",),
                        values = np.arange(0, n_max_l, dtype=np.int32).reshape(n_max_l, 1)
                    )
                )
            )

        l_max = len(vector_expansion_blocks) - 1
        vector_expansion_tmap = TensorMap(
            keys = Labels(
                names = ("l",),
                values = np.arange(0, l_max+1, dtype=np.int32).reshape(l_max+1, 1),
            ),
            blocks = vector_expansion_blocks
        )

        return vector_expansion_tmap

