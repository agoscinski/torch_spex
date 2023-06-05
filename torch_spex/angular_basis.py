import numpy as np
import torch
import sphericart.torch as sphericart


class AngularBasis(torch.nn.Module):

    def __init__(self, l_max) -> None:
        super().__init__()

        self.l_max = l_max
        self.sh_calculator = sphericart.SphericalHarmonics(l_max, normalized=True)

    def forward(self, xyz):
        sh = self.sh_calculator.compute(xyz)
        sh = [sh[:, int(l**2):int((l+1)**2)] for l in range(self.l_max+1)]
        return sh
