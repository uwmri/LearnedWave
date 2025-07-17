import numpy as np
import cupy as cp
import sigpy as sp
import random, os, h5py, logging, argparse
from random import randrange
import matplotlib.pyplot as plt

import torch
from torch.utils.tensorboard import SummaryWriter

from nufft_sigpy import NUFFT, NUFFTadjoint
from model import ResBlock, RunningAverage, BlockWiseCNN


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

nufft_forward = NUFFT.apply
nufft_adjoint = NUFFTadjoint.apply


init_type = 'poisson'
Ntrial = randrange(10000)
num_cases_train = 36
num_cases_val = 4
num_enc_fe = 5
num_enc_nc = 4
num_frame = 0
NDIM = 3
xres = 256
yres = 256
zres = 256
scale = 1.
sphere_center = (xres // 2, yres // 2, zres // 2)
x, y, z = np.ogrid[:xres, :yres, :zres]
distance = np.sqrt((x - sphere_center[0])**2 + (y - sphere_center[1])**2 + (z - sphere_center[2])**2)
mask = np.where(distance <= xres//2, 1, 0)
ksp_central = 100
shell = np.where(distance <= ksp_central, 0, mask)
shell = sp.to_device(shell, sp.Device(0))


parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--data_folder_train', type=str,
                    default=r'I:\Data\RAW_PCVIPR_CS_clean/train',
                    help='Data path')
parser.add_argument('--data_folder_val', type=str,
                    default=r'I:\Data\RAW_PCVIPR_CS_clean/val',
                    help='Data path')
parser.add_argument('--log_dir', type=str,
                    default=rf'I:\Data\run_flownoise\{Ntrial}',
                    help='Directory to log files')
parser.add_argument('--wave_dir', type=str,
                    default=r'I:\code\2Dsampling',
                    help='Directory to save learned params')
parser.add_argument('--Nepochs', type=int, default=3000)
parser.add_argument('--r', type=int, default=6)
parser.add_argument('--unroll', type=int, default=6)
parser.add_argument('--cycles', type=int, default=8)
parser.add_argument('--nro', type=int, default=1000)
parser.add_argument('--acc', type=float, default=16)
parser.add_argument('--learning_rate', type=float, default=1e-3)
parser.add_argument('--learning_rate_coords', type=float, default=1e-4)
parser.add_argument('--num_coils', type=int, default=12)
parser.add_argument('--pname', type=str, default=f'wave_{Ntrial}')
parser.add_argument('--resume_train', action='store_true', default=True)
parser.add_argument('--scale_global0', type=float, default=1.)
parser.add_argument('--Ntrial0', type=int)
parser.add_argument('--Nepoch0', type=int)
parser.add_argument('--echo_frac', type=float, default=0.75)
parser.add_argument('--fov', type=int, default=220)
parser.add_argument('--save_train_images', action='store_true', default=True)
args = parser.parse_args()

writer_train = SummaryWriter(os.path.join(args.log_dir, f'train_{Ntrial}'))
writer_val = SummaryWriter(os.path.join(args.log_dir, f'val_{Ntrial}'))
logging.basicConfig(filename=os.path.join(args.log_dir, f'flow_radius{args.r}_cycles{args.cycles}_{Ntrial}.log'),
                    filemode='w', level=logging.INFO)

logging.info(f'device={device}')
logging.info(f'{torch.cuda.is_available()}')

try:
    import setproctitle

    setproctitle.setproctitle(args.pname)
    print(f'Setting program name to {args.pname}')
except:
    print('setproctitle not installled,unavailable, or failed')


if args.resume_train:
    kyc0, kzc0 = np.loadtxt(os.path.join(args.log_dir, rf'centers_opt_3D_4385_9460_470.txt'), delimiter=',',
                            unpack=True)  # kzc, kyc

else:
    kyc0, kzc0 = np.loadtxt(rf'/mnt/PURENFS/cxt004/2Dsampling/run_cs/centers_256poisson_8556.txt', delimiter=',',
                            unpack=True)  # kzc, kyc
num_views = kyc0.shape[0]

kyc0 *= scale
kzc0 *= scale
kyc = torch.from_numpy(kyc0).cuda().flatten()
kzc = torch.from_numpy(kzc0).cuda().flatten()
kxc = torch.zeros_like(kyc)
kyc.requires_grad = True
kzc.requires_grad = True

# NOTE: readout is X on all scanner related stuff but Z on recon
helix_xread = np.loadtxt(
    os.path.join(args.wave_dir, f'helix_calibrated_r{args.r}c{args.cycles}echo{args.echo_frac}_fov{args.fov}.txt'),
    delimiter=',')
helix = helix_xread.astype(np.float32)
helix = helix[::2, :]
helix[:, 0] *= scale
helix[:, 1] *= scale
helix[:, 2] *= scale
Nsamplesz = helix.shape[0]

helixplt = plt.figure()
ax = helixplt.add_subplot(projection='3d')
ax.scatter(helix[:, 0], helix[:, 1], helix[:, 2])
writer_train.add_figure('helix', helixplt)

if args.resume_train:
    radius_scale, theta = np.loadtxt(os.path.join(args.log_dir, f'scale_rot_{args.Ntrial0}_{args.Nepoch0}.txt'), delimiter=',', unpack=True)
    radius_scale = torch.from_numpy(radius_scale).cuda()
    theta = torch.from_numpy(theta).cuda()
    radius_scale.requires_grad = False
    theta.requires_grad = False
    # get scale_global and scalek from logging file from previous run
    scale_global = torch.tensor([args.scale_global0], requires_grad=True, device='cuda')
    scalek = torch.tensor(
        [1.0510849952697754, 1.0130218267440796, 1.055649995803833, 1.0037169456481934, 1.028354287147522],
        requires_grad=True, device='cuda')

    kw0 = np.loadtxt(os.path.join(args.log_dir, f'kw_{args.Ntrial0}_{args.Nepoch0}.txt'), delimiter=',')  # kzc, kyc
    kw = torch.from_numpy(kw0).cuda()
    kw = kw.reshape((-1,))
    kw.requires_grad = True

else:
    # use gradient amplitude as kw. This file is the calibrated wave gradients
    g = np.loadtxt(
        os.path.join(args.wave_dir, f'Rwave_r{args.r}c{args.cycles}_gc_{args.echo_frac}_wimr2_fov{args.fov}.txt'),
        delimiter=',')
    g = g[::2, :]
    t = np.arange(0, g.shape[0]) * 4e-3 + 0.0
    kw = torch.from_numpy(np.sqrt(g[:, 0] ** 2 + g[:, 1] ** 2 + 0.750298 ** 2)).cuda()  #0.750298 is the mag of the trapzoidp
    kw = (kw - kw.min()) / (kw.max() - kw.min())
    kw = kw.unsqueeze(1)
    kw = kw.repeat(1, num_views)
    kw = kw.reshape((-1,))
    kw.requires_grad = True

    radius_scale = torch.ones([num_views], requires_grad=True, device='cuda')
    with torch.no_grad():
        radius_scale = radius_scale * torch.tensor([5.0]).cuda()

    theta = torch.zeros([num_views], requires_grad=True, device='cuda')

    scale_global = torch.ones([1], requires_grad=True, device='cuda')
    scalek = torch.ones([args.unroll], requires_grad=True, device='cuda')


activationC = torch.nn.Hardtanh(min_val=- xres / 2 * scale, max_val=xres / 2 * scale, inplace=False)

writer_train = SummaryWriter(os.path.join(args.log_dir, f'train_{Ntrial}'))

evalimagesfile = os.path.join(args.log_dir, f'eval_{Ntrial}.h5')
try:
    os.remove(evalimagesfile)
except OSError:
    pass

denoiser = ResBlock(1, 1, true_complex=True)
patch_size = [128, 128, 128]
overlap = [20, 20, 20]
denoiser_bw = BlockWiseCNN(denoiser, patch_size, overlap).to(device)
optimizer = torch.optim.Adam([{'params': denoiser_bw.parameters()},
                              {'params': scale_global},
                              {'params': scalek},
                              {'params': kw},
                              {'params': radius_scale},
                              {'params': theta},
                              {'params': kyc, 'lr': args.learning_rate_coords},
                              {'params': kzc, 'lr': args.learning_rate_coords}
                              ], lr=args.learning_rate, weight_decay=1e-4)
denoiser_bw.cuda()
if args.resume_train:
    ptfile = os.path.join(args.log_dir, f'denoiser_{args.Ntrial0}_{args.Nepoch0}.pt')
    state = torch.load(ptfile)
    denoiser_bw.load_state_dict(state['state_dict'], strict=True)
    optimizer.load_state_dict(state['optimizer'])


for epoch in range(args.Nepoch):

    train_avg = RunningAverage()
    eval_avg = RunningAverage()

    denoiser_bw.train()

    # data loading
    idx = np.arange(num_cases_train + num_cases_val)
    idx = np.delete(idx, [2, 14])
    idx = np.random.permutation(idx)

    for cs in range(num_cases_train):
        optimizer.zero_grad()

        # add gaussian noise
        gaussian_level = random.uniform(10, 30)

        radius_scale_clamp = torch.tanh(radius_scale)
        kyc_clamp = activationC(kyc)
        kzc_clamp = activationC(kzc)

        i = idx[cs]
        if i > 19:
            enc = np.random.randint(num_enc_fe)
        else:
            enc = np.random.randint(num_enc_nc)
        imagesfile = os.path.join(args.data_folder_train, rf'{i}/Images.h5')

        with h5py.File(imagesfile, 'r') as hf:
            truth = hf['Images'][f'Encode_{enc:03}_Frame_{num_frame:03}']
            truth = np.array(truth['real'] + 1j * truth['imag'])
        scale_truth = 1.0 / np.max(np.abs(truth))
        truth *= scale_truth
        truth = np.rot90(truth, k=1, axes=(0, 2))
        truth = np.rot90(truth, k=1, axes=(2, 1))

        truth_gpu = sp.to_device(truth, sp.Device(0))
        ktruth = cp.fft.ifftshift(truth_gpu, axes=(-3, -2, -1))
        ktruth = cp.fft.fftn(ktruth, axes=(-3, -2, -1))
        ktruth = cp.fft.fftshift(ktruth, axes=(-3, -2, -1))
        kedge = ktruth * shell
        kedge = kedge.reshape((-1,))
        kedge = kedge[np.nonzero(kedge)]
        sigma_est_real = np.median(np.abs(np.real(kedge)))
        sigma_est_imag = np.median(np.abs(np.imag(kedge)))
        sigma_real = gaussian_level * sigma_est_real
        sigma_imag = gaussian_level * sigma_est_imag

        gauss_real = cp.random.normal(0.0, sigma_real, (xres, xres, xres))
        gauss_imag = cp.random.normal(0.0, sigma_imag, (xres, xres, xres))
        knoisy_real = cp.real(ktruth) + gauss_real
        knoisy_imag = cp.imag(ktruth) + gauss_imag
        knoisy = knoisy_real + 1j * knoisy_imag
        truth_noisy = cp.fft.ifftshift(knoisy, axes=(-3, -2, -1))
        truth_noisy = cp.fft.ifftn(truth_noisy, axes=(-3, -2, -1))
        truth_noisy = cp.fft.fftshift(truth_noisy, axes=(-3, -2, -1))

        smapsfile = os.path.join(args.data_folder_train, rf'{i}/SenseMaps.h5')
        with h5py.File(smapsfile, 'r') as hf:
            smaps = []
            for cc in range(args.num_coils):
                smap = hf['Maps'][f'SenseMaps_{cc}']
                try:
                    smaps.append(np.array(smap['real'] + 1j * smap['imag']))
                except:
                    smaps.append(smap)
            smaps = np.stack(smaps, axis=0)
        smaps = np.rot90(smaps, k=1, axes=(1, 3))
        smaps = np.rot90(smaps, k=1, axes=(3, 2))

        truth_tensor = torch.tensor(truth.copy(), dtype=torch.complex64)
        truth_tensor = truth_tensor.cuda()
        truth_tensor.requires_grad = False

        truth_tensor2 = torch.tensor(truth_noisy.get().copy(), dtype=torch.complex64)
        truth_tensor2 = truth_tensor2.cuda()
        truth_tensor2.requires_grad = False

        smaps_tensor = torch.from_numpy(smaps.copy()).type(torch.complex64)
        smaps_tensor.requires_grad = False

        # generate full coordinates
        kyzc = torch.stack((kyc_clamp, kzc_clamp), axis=1)
        coords_centers = torch.stack((kxc, kyc_clamp, kzc_clamp), axis=1)
        helix_torch = torch.from_numpy(helix.copy()).cuda()
        coord = torch.zeros(helix.shape + (num_views,), dtype=torch.float32).cuda()
        for v in range(num_views):
            rot = torch.stack(
                (torch.tensor(1., device='cuda'), torch.tensor(0., device='cuda'), torch.tensor(0., device='cuda'),
                 torch.tensor(0., device='cuda'), torch.cos(theta[v]), -torch.sin(theta[v]),
                 torch.tensor(0., device='cuda'), torch.sin(theta[v]), torch.cos(theta[v])))
            rot = torch.reshape(rot, (3, 3))
            scaler = torch.stack((torch.tensor(1., device='cuda'), radius_scale_clamp[v], radius_scale_clamp[v]))
            helix_tmp = helix_torch * scaler
            coord[..., v] = torch.matmul(helix_tmp, rot) + coords_centers[v].unsqueeze(0)
        coord = torch.permute(coord, (0, -1, 1))
        coord = torch.reshape(coord, (-1, NDIM))

        cp._default_memory_pool.free_all_blocks()
        cp.get_default_pinned_memory_pool().free_all_blocks()
        torch.cuda.empty_cache()

        y = nufft_forward(coord, truth_tensor2, smaps_tensor)
        y_weighted = y * kw
        im = nufft_adjoint(coord, y_weighted, smaps_tensor)
        torch.cuda.empty_cache()

        for u in range(args.unroll):
            # Data consistency
            Ex = nufft_forward(coord, im, smaps_tensor)
            Ex = Ex / scalek[u]
            logging.info(f'train epoch {epoch}, unroll {u}, scalek = {scalek[u].item()}')
            Exd = Ex - y  # Ex-d
            Exd = Exd * kw
            im = im - nufft_adjoint(coord, Exd, smaps_tensor) * scale_global

            im = denoiser_bw(im[None, None])
            im = im.squeeze()
            torch.cuda.empty_cache()

        lossMSE = (im.squeeze() - truth_tensor.squeeze()).abs().pow(2).sum().pow(0.5) / truth_tensor.abs().pow(
            2).sum().pow(0.5)
        lossMSE.backward(retain_graph=True)
        train_avg.update(lossMSE.detach().item())
        optimizer.step()
        torch.cuda.empty_cache()

    writer_train.add_scalar('LossMSE', train_avg.avg(), epoch)
    writer_train.add_scalar('scale_global', scale_global.detach().cpu().numpy(), epoch)
    logging.info(f'Epoch {epoch} lossMSE =  {lossMSE.detach().item()}')

    # save some images, learned params and denoiser states
    if epoch % 10 == 0:
        with h5py.File(evalimagesfile, 'a') as hf:
            if epoch == 0:
                hf.create_dataset(f"truth_mag", data=np.squeeze(np.abs(truth)))
            if args.save_train_images and epoch % 10 == 0:
                hf.create_dataset(f"im_epoch{epoch}_train", data=np.squeeze(np.abs(im.detach().cpu().numpy())))

    if epoch % 10 == 0 or epoch == args.Nepoch - 1:
        rotplt, rotax = plt.subplots(figsize=(6, 6))
        rotscatter = rotax.scatter(kyc_clamp.detach().cpu().numpy() / scale, kzc_clamp.detach().cpu().numpy() / scale,
                                   c=theta.detach().cpu().numpy(), s=3, alpha=0.7)
        rotlegend = rotax.legend(*rotscatter.legend_elements(num=4),
                                 loc="upper right", title="Theta")
        rotax.add_artist(rotlegend)
        writer_train.add_figure('theta', rotplt, epoch)

        rotpltzoom, ax = plt.subplots(figsize=(6, 6))
        scatter = ax.scatter(kyc_clamp.detach().cpu().numpy() / scale, kzc_clamp.detach().cpu().numpy() / scale,
                             c=theta.detach().cpu().numpy(), s=10, alpha=0.7, cmap='Spectral')
        plt.xlim(-15, 15)
        plt.ylim(-15, 15)
        legend = ax.legend(*scatter.legend_elements(num=4),
                           loc="upper right", title="Theta")
        ax.add_artist(legend)
        writer_train.add_figure('theta_zoom', rotpltzoom, epoch)

        scalerotplt, ax = plt.subplots(figsize=(6, 6))
        scatter = ax.scatter(kyc_clamp.detach().cpu().numpy() / scale, kzc_clamp.detach().cpu().numpy() / scale,
                             c=radius_scale_clamp.detach().cpu().numpy() * args.r, s=3, alpha=0.7)
        legend = ax.legend(*scatter.legend_elements(num=4),
                           loc="upper right", title="Radius")
        ax.add_artist(legend)
        writer_train.add_figure('radius', scalerotplt, epoch)

        scalerotpltzoom, ax = plt.subplots(figsize=(6, 6))
        scatter = ax.scatter(kyc_clamp.detach().cpu().numpy() / scale, kzc_clamp.detach().cpu().numpy() / scale,
                             c=radius_scale_clamp.detach().cpu().numpy() * args.r, s=10, alpha=0.7, cmap='Spectral')
        plt.xlim(-15, 15)
        plt.ylim(-15, 15)
        legend = ax.legend(*scatter.legend_elements(num=4),
                           loc="upper right", title="Radius")
        ax.add_artist(legend)
        writer_train.add_figure('radius_zoom', scalerotpltzoom, epoch)

        final = np.stack((kyc_clamp.detach().cpu().numpy() / scale, kzc_clamp.detach().cpu().numpy() / scale), axis=1)
        coordsfile = os.path.join(args.log_dir, f'centers_opt_3D_{num_views}_{Ntrial}_{epoch}.txt')
        try:
            os.remove(coordsfile)
        except OSError:
            pass
        np.savetxt(coordsfile, final, fmt='%f', delimiter=",")

        denoiserstate = {
            'state_dict': denoiser_bw.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch
        }
        denoiserfile = os.path.join(args.log_dir, f'denoiser_{Ntrial}_{epoch}.pt')
        try:
            os.remove(denoiserfile)
        except OSError:
            pass
        torch.save(denoiserstate, denoiserfile)

        scale_global_np = scale_global.detach().cpu().numpy()
        logging.info(f'scale_global={scale_global_np}')
        scalerot = np.stack((radius_scale_clamp.detach().cpu().numpy(), theta.detach().cpu().numpy()), axis=1)
        scalerotfile = os.path.join(args.log_dir, f'scale_rot_{Ntrial}_{epoch}.txt')
        try:
            os.remove(scalerotfile)
        except OSError:
            pass
        np.savetxt(scalerotfile, scalerot, fmt='%f', delimiter=",")

        kw2d = kw.detach().clone().cpu()
        kw2d = kw2d.reshape((Nsamplesz, num_views))
        kwfile = os.path.join(args.log_dir, f'kw_{Ntrial}_{epoch}.txt')
        try:
            os.remove(kwfile)
        except OSError:
            pass
        np.savetxt(kwfile, kw2d, fmt='%f', delimiter=",")

    del im, y
    torch.cuda.empty_cache()

    with torch.no_grad():
        denoiser_bw.eval()
        idx = [2, 14]
        encs = [0, 3, 1, 4]
        for cs in range(num_cases_val):
            radius_scale_clamp = torch.tanh(radius_scale)
            kyc_clamp = activationC(kyc)
            kzc_clamp = activationC(kzc)

            i = idx[cs]
            enc = encs[cs]
            imagesfile = os.path.join(args.data_folder_val, rf'{i}/Images.h5')
            with h5py.File(imagesfile, 'r') as hf:
                truth = hf['Images'][f'Encode_{enc:03}_Frame_{num_frame:03}']
                truth = np.array(truth['real'] + 1j * truth['imag'])
            # scale_truth = 1.0 / np.max(np.abs(truth))
            # truth *= scale_truth
            truth = np.rot90(truth, k=1, axes=(0, 2))
            truth = np.rot90(truth, k=2, axes=(1, 2))

            truth_gpu = sp.to_device(truth, sp.Device(0))
            ktruth = cp.fft.ifftshift(truth_gpu, axes=(-3, -2, -1))
            ktruth = cp.fft.fftn(ktruth, axes=(-3, -2, -1))
            ktruth = cp.fft.fftshift(ktruth, axes=(-3, -2, -1))
            kedge = ktruth * shell
            kedge = kedge.reshape((-1,))
            kedge = kedge[np.nonzero(kedge)]

            sigma_est_real = np.median(np.abs(np.real(kedge)))
            sigma_est_imag = np.median(np.abs(np.imag(kedge)))
            sigma_real = gaussian_level * sigma_est_real
            sigma_imag = gaussian_level * sigma_est_imag

            gauss_real = cp.random.normal(0.0, sigma_real, (xres, xres, xres))
            gauss_imag = cp.random.normal(0.0, sigma_imag, (xres, xres, xres))
            knoisy_real = cp.real(ktruth) + gauss_real
            knoisy_imag = cp.imag(ktruth) + gauss_imag
            knoisy = knoisy_real + 1j * knoisy_imag
            truth_noisy = cp.fft.ifftshift(knoisy, axes=(-3, -2, -1))
            truth_noisy = cp.fft.ifftn(truth_noisy, axes=(-3, -2, -1))
            truth_noisy = cp.fft.fftshift(truth_noisy, axes=(-3, -2, -1))

            smapsfile = os.path.join(args.data_folder_val, rf'{i}/SenseMaps.h5')
            with h5py.File(smapsfile, 'r') as hf:
                smaps = []
                for cc in range(args.num_coils):
                    smap = hf['Maps'][f'SenseMaps_{cc}']
                    try:
                        smaps.append(np.array(smap['real'] + 1j * smap['imag']))
                    except:
                        smaps.append(smap)
                smaps = np.stack(smaps, axis=0)
            smaps = np.rot90(smaps, k=1, axes=(1, 3))
            smaps = np.rot90(smaps, k=2, axes=(2, 3))

            truth_tensor = torch.tensor(truth.copy(), dtype=torch.complex64)
            truth_tensor = truth_tensor.cuda()
            truth_tensor.requires_grad = False

            truth_tensor2 = torch.tensor(truth_noisy.get().copy(), dtype=torch.complex64)
            truth_tensor2 = truth_tensor2.cuda()
            truth_tensor2.requires_grad = False

            # truth_tensor = truth_tensor.unsqueeze(0)
            smaps_tensor = torch.from_numpy(smaps.copy()).type(torch.complex64)

            # generate full coordinates
            kyzc = torch.stack((kyc_clamp, kzc_clamp), axis=1)
            coords_centers = torch.stack((kxc, kyc_clamp, kzc_clamp), axis=1)
            helix_torch = torch.from_numpy(helix.copy()).cuda()

            coord = torch.zeros(helix.shape + (num_views,), dtype=torch.float32).cuda()
            for v in range(num_views):
                rot = torch.stack(
                    (torch.tensor(1., device='cuda'), torch.tensor(0., device='cuda'), torch.tensor(0., device='cuda'),
                     torch.tensor(0., device='cuda'), torch.cos(theta[v]), -torch.sin(theta[v]),
                     torch.tensor(0., device='cuda'), torch.sin(theta[v]), torch.cos(theta[v])))
                rot = torch.reshape(rot, (3, 3))
                scaler = torch.stack((torch.tensor(1., device='cuda'), radius_scale_clamp[v], radius_scale_clamp[v]))
                helix_tmp = helix_torch * scaler
                coord[..., v] = torch.matmul(helix_tmp, rot) + coords_centers[v].unsqueeze(0)
            coord = torch.permute(coord, (0, -1, 1))
            coord = torch.reshape(coord, (-1, NDIM))

            y = nufft_forward(coord, truth_tensor2, smaps_tensor)
            y_weighted = y * kw
            im = nufft_adjoint(coord, y_weighted, smaps_tensor)
            torch.cuda.empty_cache()

            for u in range(args.unroll):
                # DC
                Ex = nufft_forward(coord, im, smaps_tensor)
                logging.info(f'val epoch {epoch} unroll {u} scalek = {scalek[u].item()}')
                Ex = Ex / scalek[u]
                Exd = Ex - y  # Ex-d
                Exd = Exd * kw
                im = im - nufft_adjoint(coord, Exd, smaps_tensor) * scale_global

                # CNN
                im = denoiser_bw(im[None, None])
                im = im.squeeze()

                torch.cuda.empty_cache()

            lossMSE_val = (im.squeeze() - truth_tensor.squeeze()).abs().pow(2).sum().pow(0.5) / truth_tensor.abs().pow(
                2).sum().pow(0.5)

            eval_avg.update(lossMSE_val.detach().item())
            if epoch % 10 == 0 and i == 0:
                with h5py.File(evalimagesfile, 'a') as hf:
                    if epoch == 0:
                        hf.create_dataset(f"eval_truth_mag", data=np.squeeze(np.abs(truth)))

                    hf.create_dataset(f"eval_im_epoch{epoch}_mag", data=np.squeeze(np.abs(im.detach().cpu().numpy())))

            del im, y
            torch.cuda.empty_cache()

        writer_val.add_scalar('LossMSE', eval_avg.avg(), epoch)
