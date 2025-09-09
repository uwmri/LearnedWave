# this script reconstruct prospective wave/no wave data
# takes in kspace data written by get_kspace_ScanArchive.m
# Trained model and optimized centers/scale_rot 5784 is wave readout and 5242 is no-wave readout.

import argparse
from pathlib import Path

import torch
import numpy as np
import cupy as cp
import sigpy as sp
import sigpy.mri as mri
from sigpy.pytorch import to_pytorch, from_pytorch
import os, re, h5py, math
import matplotlib.pyplot as plt
from math import ceil

from src.utils import pca_coil_compression
from src.model import ResBlock, BlockWiseCNN

def parse_args():
    parser = argparse.ArgumentParser(
        description="Reconstruct prospective wave/no wave data"
    )
    parser.add_argument("--recon_type", default="train",
                        choices=["train", "l1w", "sense"])
    parser.add_argument("--Ntrial", type=int, default=5242)
    parser.add_argument("--num_unroll", type=int, default=6)
    parser.add_argument("--opt_dir", type=Path, default=Path('S:/code/LearnedWave'))
    parser.add_argument("--train_root", type=Path,
                        default=Path('S:/code/LearnedWave/trained_models'))
    parser.add_argument("--scan_root", type=Path,
                        default=Path('D:/SSD/Data/Scans_i/wave_flow_vol/v18ctang_02096_2024-06-25'))
    parser.add_argument("--scan_subdir", type=Path,
                        default=Path('02145_00016_nowave_flow/raw_data'))
    parser.add_argument('--coil_compression', action='store_true', default=False)
    parser.add_argument('--true_complex', action='store_true', default=True)


    return parser.parse_args()

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args = parse_args()
    scan_root = str(args.scan_root)
    scan_dir = str(args.scan_root / args.scan_subdir)

    # NOTE: run this code without CNN denoiser, stepsize=1/max_eig, many unrolls first, to get an estimated image and scale_im
    # then scale the raw kspace data with scale_im. Because the raw kspace scale is VERY different than training/sim
    # during training, truth was normalized by its max.
    RUN = 1
    if RUN > 0:
        run0fname = 'recon_train6_denoiserFalse_RUN0.h5'
        with h5py.File(os.path.join(scan_dir, run0fname), 'r') as hf:
            im0 = np.array(hf[f'1_IMAGE'])

        scale_im = 1.0 / np.max(np.abs(im0))
        if_denoiser = True
    else:
        scale_im = 1.0
        if_denoiser = False


    if args.coil_compression:
        num_coils = 12

    recon_type = args.recon_type
    if recon_type in ("l1w", "sense"):
        num_iter = 30

    scale_global = {5784: 0.6468, 5242: 0.51}.get(args.Ntrial, 1.0)
    Nepoch = {5784: 16, 5242: 20}.get(args.Ntrial, 1.0)

    # scan params
    num_views = 4385
    radius = 6
    helix_cycles = 8
    echo_frac = 0.75
    fov = 220
    num_enc = 4
    xres = 256
    yres = 256
    zres = 256

    # parameters from ScanArchive header. Shifts from iso
    with open(f'{scan_dir}/header.txt', 'r') as file:
        text = file.read()
    rhuser24 = float(re.search(r'rhuser24\s*=\s*([-+]?\d*\.?\d+)', text).group(1))
    rhuser25 = float(re.search(r'rhuser25\s*=\s*([-+]?\d*\.?\d+)', text).group(1))
    rhuser26 = float(re.search(r'rhuser26\s*=\s*([-+]?\d*\.?\d+)', text).group(1))
    oc_xshift = -rhuser24
    oc_yshift = -rhuser25
    oc_zshift = -rhuser26
    deltax = 2 * math.pi * oc_xshift / fov
    deltay = 2 * math.pi * oc_yshift / fov
    deltaz = 2 * math.pi * oc_zshift / fov

    opt_dir = str(args.opt_dir)
    train_dir = str(args.train_root / f"{args.Ntrial}")
    centersfile = f"centers_opt_3D_{num_views}_{args.Ntrial}_{Nepoch}_permute.txt"
    scalerotfile = f"scale_rot_{args.Ntrial}_{Nepoch}_permute.txt"
    helix_calib = np.loadtxt(
        os.path.join(opt_dir, f"helix_calibrated_r{radius}c{helix_cycles}echo{echo_frac}_fov{fov}.txt"),
        delimiter=",",
    )
    res_gHelix = helix_calib.shape[0]
    helix = helix_calib[::2, :]
    if args.Ntrial == 5242:
        helix[:, 1] = 0.0
        helix[:, 2] = 0.0

    # use lowres smaps from a separate scan
    hfsmaps = h5py.File(name=os.path.join(scan_root, f'lowres_fov{fov}_ax_shifted.h5'), mode='r')
    smaps = hfsmaps['lowres_coil'][()]

    if if_denoiser:
        denoiser_file = os.path.join(train_dir, f'denoiser_{args.Ntrial}_{Nepoch}.pt')
        state = torch.load(denoiser_file)
        denoiser = ResBlock(1, 1, true_complex=args.true_complex)
        patch_size = [128, 128, 128]
        overlap = [20, 20, 20]
        denoiser_bw = BlockWiseCNN(denoiser, patch_size, overlap).to(device)
        denoiser_bw.load_state_dict(state['state_dict'], strict=True)
        denoiser_bw.cuda()
        denoiser_bw.eval()


    # get full coords
    radius_scale, theta = np.loadtxt(os.path.join(train_dir, scalerotfile), delimiter=',', unpack=True)
    kyzc_raw = np.loadtxt(os.path.join(train_dir, centersfile), delimiter=',')  # kzc, kyc
    kyzc = kyzc_raw[:num_views, :]
    centers = np.zeros((num_views, 3), dtype=kyzc.dtype)
    centers[:, 1:] = kyzc   # x is readout
    helix_interp = helix
    coords = np.zeros(helix_interp.shape + (num_views,), dtype=np.float32)
    for v in range(num_views):
        rot = np.stack((1., 0., 0.,
                        0., np.cos(theta[v]), -np.sin(theta[v]),
                        0., np.sin(theta[v]), np.cos(theta[v])))
        rot = np.reshape(rot, (3, 3))
        scaler = np.stack((1., radius_scale[v], radius_scale[v]))
        helix_tmp = helix_interp * scaler
        coords[..., v] = np.matmul(helix_tmp, rot) + np.expand_dims(centers[v], axis=0)
    coords = np.transpose(coords, (0, -1, 1))
    coord3D = coords.copy()
    coord = np.reshape(coord3D, (-1, 3))
    coord = sp.to_device(coord, sp.Device(0))

    if recon_type == 'train':
        recondelayfile = os.path.join(scan_dir, f'recon_{recon_type}{args.num_unroll}_denoiser{if_denoiser}.h5')
    else:
        recondelayfile = os.path.join(scan_dir, f'recon_{recon_type}.h5')
    try:
        os.remove(recondelayfile)
    except OSError:
        pass

    # do recon
    imAll = []
    for enc in range(num_enc):
        hf = h5py.File(name=os.path.join(scan_dir, f'raw{enc + 1}{num_enc}.h5'), mode='r')
        kdata_raw = hf['kdata_r'][()].astype(np.complex64) + 1j * hf['kdata_i'][()].astype(np.complex64)
        kdata = kdata_raw[:, :, :num_views]
        kdata = kdata[:, ::2, :]

        # occasionally, last 4 coil elements were turned off
        num_coils0 = kdata.shape[0]
        smaps_ = smaps[:num_coils0, ...]

        kdata_cc_shift = kdata * np.exp(1j * coord3D[:, :, 0] * deltax)
        kdata_cc_shift = kdata_cc_shift * np.exp(1j * coord3D[:, :, 1] * deltay)
        kdata_cc_shift = kdata_cc_shift * np.exp(1j * coord3D[:, :, 2] * deltaz)
        k = np.reshape(kdata_cc_shift, (num_coils0, -1))
        if args.coil_compression:
            k = pca_coil_compression(kdata=k, axis=0, target_channels=num_coils)

        if recon_type == 'train':
            k_gpu = sp.to_device(k * scale_im, sp.Device(0))
            A = sp.mri.linop.Sense(smaps_, coord=coord, weights=None, tseg=None,
                                   coil_batch_size=1, comm=None,
                                   transp_nufft=False)

            im = torch.zeros([xres, yres, zres], dtype=torch.complex64, device='cuda')
            im.requires_grad = False
            if RUN == 0 and enc == 0:
                max_eig = sp.app.MaxEig(A.H * A, max_iter=30, dtype=np.complex64, device=sp.Device(0),
                                        show_pbar=True).run()
            for u in range(args.num_unroll):
                Ex = A * from_pytorch(im)
                Exd = Ex - k_gpu  # Ex-d
                if RUN == 0:
                    im = im - to_pytorch(A.H * Exd) * 1 / max_eig
                else:
                    im = im - to_pytorch(A.H * Exd) * scale_global
                torch.cuda.empty_cache()
                im = im.detach()

                im = denoiser_bw(im[None, None])
                im = im.squeeze()
                torch.cuda.empty_cache()
            im = im.detach().cpu().numpy()
        elif recon_type == 'l1w':
            k_gpu = sp.to_device(k, sp.Device(0))
            im = sp.mri.app.L1WaveletRecon(k_gpu, mps=smaps_, weights=None, coord=coord,
                                           device=sp.Device(0), lamda=1e-3, max_iter=num_iter,
                                           coil_batch_size=1, save_objective_values=False).run()
        elif recon_type == 'sense':
            k_gpu = sp.to_device(k, sp.Device(0))
            im = sp.mri.app.SenseRecon(k_gpu, mps=smaps_, weights=None, coord=coord,
                                       device=sp.Device(0), lamda=1e-3, max_iter=num_iter,
                                       coil_batch_size=1, save_objective_values=False).run()
        im = np.squeeze(im.get().astype(np.complex64))
        imAll.append(im)
        del im, k_gpu
        torch.cuda.empty_cache()
        cp._default_memory_pool.free_all_blocks()
    imAll = np.asarray(imAll).astype(np.complex64)
    imAll = np.flip(np.transpose(imAll, (0, 3, 2, 1)), (1, 2, 3))
    try:
        os.remove(recondelayfile)
    except OSError:
        pass
    with h5py.File(recondelayfile, 'a') as hf:
        hf.create_dataset(f"1_IMAGE_mag", data=np.squeeze(np.abs(imAll)))
        hf.create_dataset(f"1_IMAGE_ph", data=np.squeeze(np.angle(imAll)))
        hf.create_dataset(f"1_IMAGE", data=imAll)


if __name__ == "__main__":
    main()