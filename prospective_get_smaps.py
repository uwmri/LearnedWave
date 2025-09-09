"""
    This script is used for getting smaps from separate smaps scan for pcvipr-wave scans
"""

import torch
print(torch.cuda.is_available())
print(torch.backends.cudnn.is_available())

import numpy as np
import math
import sigpy as sp
from sigpy import backend
from sigpy.fourier import _scale_coord, _apodize, _get_oversamp_shape
import h5py
import os
from src.utils import pca_coil_compression, hann
import cupy as cp

scan_dir = r'D:\SSD\Data\Scans_i\wave_flow_vol\v22zyardim_2024-07-06\02145_00014_smaps32\raw_data'
scan_root = rf'D:\SSD\Data\Scans_i\wave_flow_vol\v22zyardim_2024-07-06'
opt_dir = r'S:\code\2Dsampling'

hf = h5py.File(name=os.path.join(scan_dir, 'raw.h5'), mode='r')
kdata_raw = (hf['kdata_r'][()].astype(np.float32) +1j *hf['kdata_i'][()].astype(np.float32))
kdata = kdata_raw
num_coils = kdata.shape[0]
res_gHelix = kdata.shape[1]

orientation = 'ax'
coil_compression = False
target_coils = 12
zero_pad = False

# scan params
xres = 256
yres = 256
zres = 256
acs = 32
echo_frac = 0.75
gamma = 4257.59    # Hz/G
pw_gxw = res_gHelix * 4
fov = 220
image_size = (xres, yres, zres)
num_views = acs * acs

# gridding params
width = 4
oversamp = 1.25
beta = np.pi * (((width / oversamp) * (oversamp - 0.5)) ** 2 - 0.8) ** 0.5

# coords
a_gxw = 1e7 * echo_frac / (gamma * (fov/acs)) / pw_gxw
kxmax = 1e-7 * fov * gamma * pw_gxw * a_gxw
kx = np.linspace(-kxmax * (echo_frac-.5)/echo_frac, kxmax * (.5/echo_frac), res_gHelix)
ky = np.zeros_like(kx)
kz = np.zeros_like(kx)
helix = np.stack((kx, ky, kz), axis=1)  #x is readout
nro = helix.shape[0]
# center coords
kyzc = np.loadtxt(os.path.join(opt_dir, 'centers_32cart.txt'), delimiter=',')  # kzc, kyc
centers = np.zeros((num_views, 3), dtype=kyzc.dtype)
centers[:, 1:] = kyzc

coord = np.zeros(helix.shape + (num_views,), dtype=np.float32)
for v in range(num_views):
    coord[..., v] = helix + np.expand_dims(centers[v], axis=0)
coord = np.transpose(coord, (0, -1, 1))
coord3D = coord.copy()
coord = np.reshape(coord, (-1, 3))
coord = sp.to_device(coord, sp.Device(0))   # (num_views*num_pts, 3)

# shift from isocenter
rhuser24 = 0.0
rhuser25 = -28.12
rhuser26 = 3.12
oc_xshift = -rhuser24
oc_yshift = -rhuser25
oc_zshift = -rhuser26
deltax = 2*math.pi * oc_xshift/fov
deltay = 2*math.pi * oc_yshift/fov
deltaz = 2*math.pi * oc_zshift/fov
# for Axial, freq. enc. AP
kdata_cc_shift = kdata*np.exp(1j*coord3D[:,:,0]*deltax)
kdata_cc_shift = kdata_cc_shift*np.exp(1j*coord3D[:,:,1]*deltay)
kdata_cc_shift = kdata_cc_shift*np.exp(1j*coord3D[:,:,2]*deltaz)
if coil_compression:
    num_coils = 12
    kdata_cc_shift = pca_coil_compression(kdata=kdata_cc_shift.get(), axis=0, target_channels=num_coils)
    kdata_cc_shift = sp.to_device(kdata_cc_shift, device=sp.Device(0))

k = np.reshape(kdata_cc_shift,(num_coils,-1))
k_gpu = sp.to_device(k, sp.Device(0))

# hann window mask
[kxx, kyy, kzz] = np.meshgrid(np.linspace(-acs//2, acs//2, acs), np.linspace(-acs//2, acs//2, acs), np.linspace(-acs//2, acs//2, acs), indexing='ij')
kr = (kxx**2 + kyy**2 + kzz**2) ** .5
fullwidth = np.sqrt(acs**2 *2)
window = hann(kr, fullwidth)
mask = np.zeros(image_size, dtype=np.float32)
idxL = (xres-window.shape[0])//2
idxS = (yres-window.shape[1])//2
idxA = (zres-window.shape[2])//2
mask[idxL:idxL+acs, idxS:idxS+acs,idxA:idxA+acs] = window
mask = sp.to_device(mask, sp.Device(0))


# Gridding recon
ndim = coord.shape[-1]
oshape = (acs, acs, acs)
os_shape = _get_oversamp_shape(oshape, ndim, oversamp)
coord_scale = _scale_coord(coord, oshape, oversamp)
im_coils = np.zeros((num_coils,)+oshape, dtype=np.complex64)
im_coils_lowres = np.zeros((num_coils,)+image_size, dtype=np.complex64)
for cc in range(num_coils):
    input = sp.to_device(kdata_cc_shift[cc], sp.Device(0))
    input = np.reshape(input, (-1,))
    input = sp.to_device(input, sp.Device(0))

    output = sp.interp.gridding(input, coord_scale, os_shape,
                                kernel='kaiser_bessel', width=width, param=beta)
    output /= width ** ndim

    output = sp.fourier.ifft(output, axes=range(-ndim, 0), norm=None)
    output = sp.util.resize(output, oshape)
    output *= sp.util.prod(os_shape[-ndim:]) / sp.util.prod(oshape[-ndim:]) ** 0.5

    # Apodize
    _apodize(output, ndim, oversamp, width, beta)
    im_coils[cc] = output.get()

    kcart = sp.fourier.fft(output, center=True)

    # center the kspace
    xp = backend.get_array_module(kcart)
    kcart = xp.roll(kcart, 6, axis=2)
    kcart = sp.util.resize(kcart, list(image_size))

    # Windowing
    kcartU = kcart * mask

    im_lowres = sp.fourier.ifft(kcartU, center=True)
    im_coils_lowres[cc] = im_lowres.get()

    del input, output, kcart, kcartU, im_lowres
    torch.cuda.empty_cache()
    cp._default_memory_pool.free_all_blocks()
im_coils = im_coils.astype(np.complex64)


# sag 3dfse smaps(acquired with spgr pcvipr.e)
if (orientation=='sag' or orientation=='cor'):
    im_coils_lowres = im_coils_lowres[...,::-1]

im_lowres_sos = np.sqrt(np.sum(np.abs(im_coils_lowres)**2, axis=0))
smaps_lowres = im_coils_lowres / im_lowres_sos


coilimagesfile = os.path.join(scan_root, f'lowres_fov{fov}_{orientation}_shifted.h5')
try:
    os.remove(coilimagesfile)
except OSError:
    pass
with h5py.File(coilimagesfile, 'a') as hf:
    hf.create_dataset(f"lowres_coil_sos", data=np.abs(im_lowres_sos))
    hf.create_dataset(f"lowres_coil", data=smaps_lowres)
    hf.create_dataset(f"lowres_coil_mag", data=np.abs(smaps_lowres[0,...]))
    hf.create_dataset(f"lowres_coil_ph", data=np.angle(smaps_lowres[0,...]))

