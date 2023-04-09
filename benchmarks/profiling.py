import numpy as np
import scipy as sp
import torch
import ase.io
from torch_spex.le import Jn_zeros
from torch_spex.spherical_expansions import SphericalExpansion
from torch_spex.structures import Structures

import cProfile

device = "cpu"
print(f"Running on {device}")

torch.set_default_dtype(torch.float64)

a = 6.0
E_max = 200

structures = ase.io.read("../datasets/rmd17/ethanol1.extxyz", ":20")

hypers_spherical_expansion = {
    "cutoff radius": 6.0,
    "radial basis": {
        "r_cut": 6.0,
        "E_max": 200 
    }
}
calculator = SphericalExpansion(hypers_spherical_expansion, [1, 6, 8], device=device)
transformed_structures = Structures(structures)
transformed_structures.to(device)

def run():
    for _ in range(100):
        spherical_expansion_coefficients_torch_spex = calculator(transformed_structures)

cProfile.runctx('run()', globals(), {'run': run}, 'profile')


import pstats
stats = pstats.Stats('profile')
stats.strip_dirs().sort_stats('tottime').print_stats(100)

import os
os.remove('profile')
