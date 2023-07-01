import numpy as np
import torch
from typing import Dict, List
import ase

from equistore import TensorMap, TensorBlock, Labels

def ase_atoms_to_tensordict(atoms_list : List[ase.Atoms], device : torch.device = None) -> Dict[str, torch.Tensor]:
    """
    dictionary contains
    - **n_structures**: ...,
    - **positions**: ...,
    - **cells**: ...,
    - **structure_indices**: ...,
    - **atomic_species**: ...
    - **cells**: ...,
    - **structure_indices**: ...,
    - **atomic_species**: ...,
    - **pbc**: ...
    """

    atomic_structures = {}

    n_total_atoms = sum([len(atoms) for atoms in atoms_list])
    n_structures = len(atoms_list)
    structures_offsets = np.cumsum([0] + [len(atoms) for atoms in atoms_list])
    atomic_structures["positions"] = torch.empty((n_total_atoms, 3), dtype=torch.get_default_dtype(), device=device)
    atomic_structures["structure_indices"] = torch.empty((n_total_atoms,), dtype=torch.int32)
    atomic_structures["atomic_species"] = torch.empty((n_total_atoms,), dtype=torch.int32)
    atomic_structures["cells"] = torch.empty((n_structures, 3, 3), dtype=torch.get_default_dtype())
    atomic_structures["pbcs"] = torch.empty((n_structures,3), dtype=torch.bool)

    for structure_index, atoms in enumerate(atoms_list):
        atoms_slice = slice(structures_offsets[structure_index], structures_offsets[structure_index+1])
        atomic_structures["positions"][atoms_slice] = torch.tensor(atoms.positions, dtype=torch.get_default_dtype(), device=device)
        atomic_structures["structure_indices"][atoms_slice] = structure_index
        atomic_structures["atomic_species"][atoms_slice] = torch.tensor(atoms.get_atomic_numbers(), dtype=torch.int32)
        atomic_structures["cells"][structure_index] = torch.tensor(atoms.cell.array, dtype=torch.get_default_dtype())
        atomic_structures["pbcs"][structure_index] = torch.tensor(atoms.pbc, dtype=torch.bool)

    atomic_structures["n_structures"] = torch.tensor(n_structures, dtype=torch.int32)
    return atomic_structures

def ase_atoms_to_tensordict_nl(atoms_list : List[ase.Atoms], cutoff: float, device : torch.device = None) -> Dict[str, torch.Tensor]:
    atomic_structures = ase_atoms_to_tensordict(atoms_list, device)
    atomic_structures["cartesian_vectors"] = get_cartesian_vectors(atomic_structures, cutoff)
    return atomic_structures

def get_cartesian_vectors(structures: Dict[str, torch.Tensor], cutoff_radius: float):

    labels = []
    vectors = []

    for structure_index in range(structures["n_structures"]):

        where_selected_structure = np.where(structures["structure_indices"] == structure_index)[0]

        centers, neighbors, unit_cell_shift_vectors = get_neighbor_list(
            structures["positions"].detach().cpu().numpy()[where_selected_structure], 
            structures["pbcs"][structure_index], 
            structures["cells"][structure_index], 
            cutoff_radius) 
        
        atoms_idx = torch.LongTensor(where_selected_structure)
        positions = structures["positions"][atoms_idx]
        cell = torch.tensor(np.array(structures["cells"][structure_index]), dtype=torch.get_default_dtype())
        species = structures["atomic_species"][atoms_idx]

        structure_vectors = positions[neighbors] - positions[centers] + (unit_cell_shift_vectors @ cell).to(positions.device)  # Warning: it works but in a weird way when there is no cell
        vectors.append(structure_vectors)
        labels.append(
            np.stack([
                np.array([structure_index]*len(centers)), 
                centers.numpy(), 
                neighbors.numpy(), 
                species[centers], 
                species[neighbors],
                unit_cell_shift_vectors[:, 0],
                unit_cell_shift_vectors[:, 1],
                unit_cell_shift_vectors[:, 2]
            ], axis=-1))

    vectors = torch.cat(vectors, dim=0)
    labels = np.concatenate(labels, axis=0)
    
    block = TensorBlock(
        values = vectors.unsqueeze(dim=-1),
        samples = Labels(
            names = ["structure", "center", "neighbor", "species_center", "species_neighbor", "cell_x", "cell_y", "cell_z"],
            values = np.array(labels, dtype=np.int32)
        ),
        components = [
            Labels(
                names = ["cartesian_dimension"],
                values = np.array([-1, 0, 1], dtype=np.int32).reshape((-1, 1))
            )
        ],
        properties = Labels.single()
    )

    return block 


def get_neighbor_list(positions, pbc, cell, cutoff_radius):

    centers, neighbors, unit_cell_shift_vectors = ase.neighborlist.primitive_neighbor_list(
        quantities="ijS",
        pbc=pbc,
        cell=cell,
        positions=positions,
        cutoff=cutoff_radius,
        self_interaction=True,
        use_scaled_positions=False,
    )

    pairs_to_throw = np.logical_and(centers == neighbors, np.all(unit_cell_shift_vectors == 0, axis=1))
    pairs_to_keep = np.logical_not(pairs_to_throw)

    centers = centers[pairs_to_keep]
    neighbors = neighbors[pairs_to_keep]
    unit_cell_shift_vectors = unit_cell_shift_vectors[pairs_to_keep]

    centers = torch.LongTensor(centers)
    neighbors = torch.LongTensor(neighbors)
    unit_cell_shift_vectors = torch.tensor(unit_cell_shift_vectors, dtype=torch.get_default_dtype())

    return centers, neighbors, unit_cell_shift_vectors
