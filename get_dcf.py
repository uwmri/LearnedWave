import matplotlib.pyplot as plt
import numpy as np
import torch
import sigpy as sp
import sigpy.mri as mri
from src.utils_torch import load_helix, load_initial_coordinates, get_full_coords

log_dir = r'S:\code\LearnedWave'
resume_train = True
Ntrial0 = 5784
Nepoch0 = 16
scale_global0 = 0.6468
radius = 6
cycles = 8
fov = 220
res = 256
Ncoils = 8
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
kyc, kzc, kxc, radius_scale, theta, scale_global = load_initial_coordinates(log_dir, resume_train, Ntrial0, Nepoch0,
                                                                            scale_global0, device, requires_grad=False)
helix = load_helix(log_dir, radius, cycles, fov, echo_frac=0.75)
coords = get_full_coords(kxc, kyc, kzc, helix, theta, radius_scale)

smaps = torch.ones((Ncoils, res, res, res), dtype=torch.complex64)
weights = torch.ones((Ncoils, coords.shape[0])).to(torch.complex64)


smaps_sp = sp.to_device(sp.from_pytorch(smaps), sp.Device(0))
coord_sp = sp.to_device(sp.from_pytorch(coords), sp.Device(0))
weights_sp = sp.to_device(sp.from_pytorch(weights), sp.Device(0))
sense_op = mri.linop.Sense(smaps_sp, coord=coord_sp, weights=None, coil_batch_size=1)
sense_adjoint = sense_op.H

for i in range(1):
    Ahw = sense_adjoint.apply(weights_sp)

    AAhw = sense_op.apply( Ahw)


    weights_sp = weights_sp /  AAhw




