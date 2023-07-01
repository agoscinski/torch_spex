def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    # Set the model to training mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        # Compute prediction and loss
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

import torch
import numpy as np
import ase.io

from torch_spex.structures import ase_atoms_to_tensordict
from torch_spex.spherical_expansions import SphericalExpansion
hypers = {
    "cutoff radius": 6,
    "radial basis": {
        "E_max": 300
    },
}
spherical_expansion = SphericalExpansion(hypers, [1,6], device="cpu")


frame = ase.io.read('inversion_fails.extxyz', 0)
frame_shifted = ase.io.read('inversion_fails.extxyz', 1)
frame = ase.io.read('../datasets/random-ch4-10k.extxyz', '0')
frame_shifted = frame.copy()
frame_shifted.positions += np.random.normal(scale=0.1, size=frame_shifted.positions.shape)

atomic_structures = ase_atoms_to_tensordict([frame])
atomic_structures_shifted = ase_atoms_to_tensordict([frame_shifted])
with torch.no_grad():
    ref = spherical_expansion.forward(atomic_structures)

def loss_fn(pred, ref):
    loss = torch.tensor(0.)
    for key in ref.keys:
        loss += torch.linalg.norm(ref.block(key).values - pred.block(key).values)
    # C env l=0
    #loss += torch.linalg.norm(ref.block(ref.keys[0]).values - pred.block(ref.keys[0]).values)
    # l=0
    #for key in [ref.keys[0], ref.keys[1]]:
    #    loss += torch.linalg.norm(ref.block(key).values - pred.block(key).values)
    return loss

atomic_structures_shifted['positions'].requires_grad_()
learning_rate = 1e-3
optimizer = torch.optim.Adam([atomic_structures_shifted['positions']], lr=learning_rate)

for step in range(200):
    # Compute prediction and loss
    pred = spherical_expansion.forward(atomic_structures_shifted)
    loss = loss_fn(pred, ref)

    # Backpropagation
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    if step % 10 == 0:
        dist = torch.linalg.norm(atomic_structures_shifted['positions'] - atomic_structures['positions'])
        print(f"loss: {loss:>7f}, actual dist: {dist} [step {step}]")

# does not work
#atomic_structures_shifted['positions'].requires_grad_()
#optimizer = torch.optim.AdamW([atomic_structures_shifted['positions']], lr=learning_rate)
#
#for step in range(10):
#    def single_step():
#        # Compute prediction and loss
#        optimizer.zero_grad()
#        pred = spherical_expansion.forward(atomic_structures_shifted)
#        pred = pred.block(0).values
#        loss = loss_fn(pred, ref)
#        # Backpropagation
#        loss.backward()
#        return loss
#    optimizer.step(single_step)
#
#    if step % 5 == 0:
#        print(f"loss: {loss:>7f}  [step {step}]")
