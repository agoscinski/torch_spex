import numpy as np
import torch
from typing import Dict, List
import ase

def ase_atoms_to_tensordict(atoms_list : List[ase.Atoms]) -> Dict[str, torch.Tensor]:
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

    positions = []
    cells = []
    structure_indices = []
    atomic_species = []
    pbcs = []

    for structure_index, atoms in enumerate(atoms_list):
        positions.append(atoms.positions)
        cells.append(atoms.cell)
        for _ in range(atoms.positions.shape[0]):
            structure_indices.append(structure_index)
        atomic_species.append(atoms.get_atomic_numbers())
        pbcs.append(atoms.pbc)

    atomic_structures = {}
    atomic_structures["n_structures"] = torch.tensor(len(atoms_list))
    atomic_structures["positions"] = torch.tensor(np.concatenate(positions, axis=0), dtype=torch.get_default_dtype())
    atomic_structures["cells"] = torch.tensor(cells)
    atomic_structures["structure_indices"] = torch.tensor(structure_indices)
    atomic_structures["atomic_species"] = torch.tensor(atomic_species)
    atomic_structures["pbcs"] = torch.tensor(pbcs)
    return atomic_structures
