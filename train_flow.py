"""
Usage
-----
python train_flow.py --config path/to/train_config.yaml
"""
import setproctitle
import argparse
import logging
import os
import math
import random
from dataclasses import dataclass, field
import yaml
from pathlib import Path
from random import randrange
import matplotlib.pyplot as plt
import h5py
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from src.model import ResBlock, RunningAverage, BlockWiseCNN
from src.nufft_sigpy import NUFFT, NUFFTadjoint
from src.utils import add_gaussian_noise, clear_gpu_mem

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NDIM = 3
XRES = 256
YRES = 256
ZRES = 256
num_cases_train = 36
num_cases_val = 4
num_enc_fe = 5
num_enc_nc = 4
num_frame = 0
echo_frac: float = 0.75
fov: int = 220

@dataclass
class TrainConfig:
    data_folder_train: str
    data_folder_val: str
    log_dir: str = "/mnt/PURENFS/cxt004/2Dsampling/run_cs"
    wave_dir: str = "."
    n_epochs: int = 3000
    r: int = 6
    cycles: int = 8
    nro: int = 1000
    acc: float = 16.0
    unroll: int = 6
    patch_size: list = field(default_factory=lambda: [128, 128, 128])
    overlap: list = field(default_factory=lambda: [20, 20, 20])
    in_channel: int = 1
    out_channel: int = 1
    true_complex: bool = True
    fmaps: int = 16
    learning_rate: float = 1e-3
    learning_rate_coords: float = 1e-4
    weight_decay: float = 1e-4
    num_coils: int = 12
    pname: str = "wave"
    resume_train: bool = False
    Ntrial0: int = 1111
    Nepoch0: int = 10
    scale_global0: float = 1.0
    gaussian_lower = 0.5
    gaussian_uppper = 1.5



def load_config(path: str) -> TrainConfig:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return TrainConfig(**data)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="learned flow: training")
    parser.add_argument(
        "--config", type=str, required=True, help="Path to YAML configuration file"
    )
    return parser.parse_args()


def setup_logging(log_dir: str, trainingID) -> None:
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        filename=os.path.join(log_dir, f"train{trainingID}.log"),
        filemode="w",
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )


class FlowTrainer:

    def __init__(self, cfg: TrainConfig, trainingID: int) -> None:
        self.cfg = cfg
        self.device = DEVICE
        self.Ntrial = trainingID
        self.writer_train = SummaryWriter(os.path.join(cfg.log_dir, f"train_{trainingID}"))
        self.writer_val = SummaryWriter(os.path.join(cfg.log_dir, f"val_{trainingID}"))
        self.evalimagesfile = os.path.join(cfg.log_dir, f'eval_{trainingID}.h5')

        self.nufft_forward = NUFFT.apply
        self.nufft_adjoint = NUFFTadjoint.apply

        self._load_initial_coordinates()
        self._load_helix()
        self._init_model()

    def _load_initial_coordinates(self) -> None:
        path = (
            os.path.join(self.cfg.log_dir, f"centers_opt_3D_4385_{self.cfg.Ntrial0}_{self.cfg.Nepoch0}.txt")
            if self.cfg.resume_train
            else os.path.join(self.cfg.wave_dir, "centers_256poisson_8556.txt")
        )
        kyc0, kzc0 = np.loadtxt(path, delimiter=",", unpack=True)
        self.num_views = kyc0.shape[0]
        self.kyc = torch.from_numpy(kyc0).to(self.device).flatten().requires_grad_(True)
        self.kzc = torch.from_numpy(kzc0).to(self.device).flatten().requires_grad_(True)
        self.kxc = torch.zeros_like(self.kyc)

        if self.cfg.resume_train:
            self.radius_scale, self.theta = np.loadtxt(os.path.join(
                self.cfg.log_dir, f'scale_rot_{self.cfg.Ntrial0}_{self.cfg.Nepoch0}.txt'),
                delimiter=',', unpack=True)
            self.radius_scale = torch.from_numpy(self.radius_scale).cuda()
            with torch.no_grad():
                self.radius_scale = torch.atanh(self.radius_scale)
            self.theta = torch.from_numpy(self.theta).cuda()
            self.radius_scale.requires_grad = True
            self.theta.requires_grad = True
            self.scale_global = torch.tensor([self.cfg.scale_global0], requires_grad=True, device='cuda')
        else:
            self.radius_scale = torch.ones([self.num_views], requires_grad=True, device='cuda')
            self.theta = torch.zeros([self.num_views], requires_grad=True, device='cuda')
            self.scale_global = torch.ones([1], requires_grad=True, device='cuda')

    def _load_helix(self) -> None:
        # requires helix trajectory file.
        # generate on host -> measure the actual trajectory
        fname = (
            f"helix_calibrated_r{self.cfg.r}c{self.cfg.cycles}"
            f"echo{self.cfg.echo_frac}_fov{self.cfg.fov}.txt"
        )
        path = os.path.join(self.cfg.wave_dir, fname)
        self.helix = np.loadtxt(path, delimiter=",").astype(np.float32)
        self.helix = self.helix[::2, :]
        self.Nsamplesz = self.helix.shape[0]

        helixplt = plt.figure()
        ax = helixplt.add_subplot(projection='3d')
        ax.scatter(self.helix[:, 0], self.helix[:, 1], self.helix[:, 2])
        self.writer_train.add_figure('helix', helixplt)

    def _init_model(self) -> None:
        self.denoiser = ResBlock(self.cfg.in_channel, self.cfg.out_channel, self.cfg.fmaps,
                                 true_complex=self.cfg.true_complex)
        self.denoiser_bw = BlockWiseCNN(self.denoiser, self.cfg.patch_size, self.cfg.overlap).to(self.device)
        self.optimizer = torch.optim.Adam(
            [{'params': self.denoiser_bw.parameters()},
             {'params': self.scale_global},
             {'params': self.radius_scale},
             {'params': self.theta},
             {'params': self.kyc, 'lr': self.cfg.learning_rate_coords},
             {'params': self.kzc, 'lr': self.cfg.learning_rate_coords}
             ], lr=self.cfg.learning_rate, weight_decay=self.cfg.weight_decay
        )

        self.activationC = torch.nn.Hardtanh(min_val=-XRES / 2, max_val=XRES / 2, inplace=False)

        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min',
                                                                    factor=0.1, patience=5, threshold=1e-4,
                                                                    threshold_mode='abs')

        if self.cfg.resume_train:
            ptfile = os.path.join(self.cfg.log_dir, f'denoiser_{self.cfg.Ntrial0}_{self.cfg.Nepoch0}.pt')
            state = torch.load(ptfile)
            self.denoiser_bw.load_state_dict(state['state_dict'], strict=True)
            self.optimizer.load_state_dict(state['optimizer'])

    def _load_truth(self, file, enc) -> np.ndarray:
        # load images reconstructed by pcvipr_recon. Normalized to its max.
        with h5py.File(file, 'r') as hf:
            truth = hf['Images'][f'Encode_{enc:03}_Frame_{num_frame:03}']
            truth = np.array(truth['real'] + 1j * truth['imag'])
        scale_truth = 1.0 / np.max(np.abs(truth))
        truth *= scale_truth
        truth = np.rot90(truth, k=1, axes=(0, 2))
        truth = np.rot90(truth, k=2, axes=(1, 2))
        return truth

    def _add_phase(self, image, amount)-> np.ndarray:
        x, y, z = np.meshgrid(np.linspace(-1, 1, 256), np.linspace(-1, 1, 256), np.linspace(-1, 1, 256), indexing='ij')

        a7 = random.uniform(-10, 10)
        a8 = random.uniform(-10, 10)
        a9 = random.uniform(-10, 10)
        a10 = random.uniform(-10, 10)
        logging.info(f'poly phase {a7}, {a8}, {a9}, {a10}')

        # poly =  a1*x**2+ a2*y**2+ a3*z**2 + a4*x*y + a5*x*z + a6*y*z + a7*x + a8*y + a9*z + a10
        poly = a7 * x + a8 * y + a9 * z + a10
        if poly.max() != 0:
            poly /= poly.max()
        image *= np.exp(1j*2 * math.pi *poly * amount)
        return image


    def _load_smaps(self, file):
        # these smaps should be from pcvipr_recon. if use other smaps, check the orientation.
        with h5py.File(file, 'r') as hf:
            smaps = []
            for cc in range(self.cfg.num_coils):
                smap = hf['Maps'][f'SenseMaps_{cc}']
                try:
                    smaps.append(np.array(smap['real'] + 1j * smap['imag']))
                except:
                    smaps.append(smap)
            smaps = np.stack(smaps, axis=0)
        smaps = np.rot90(smaps, k=1, axes=(1, 3))
        smaps = np.rot90(smaps, k=2, axes=(2, 3))
        return smaps

    def _get_kw(self, num_views):
        # currently, not learning kw. Using readout gradient magnitude as kw.
        g = np.loadtxt(
            os.path.join(self.cfg.wave_dir, f'Rwave_r{self.cfg.r}c{self.cfg.cycles}_gc_{echo_frac}_fov{fov}.txt'),
            delimiter=',')
        g = g[::2, :]
        kw = torch.from_numpy(np.sqrt(g[:, 0] ** 2 + g[:, 1] ** 2 + 0.750298 ** 2)).cuda()
        kw = (kw - kw.min()) / (kw.max() - kw.min())
        kw = kw.unsqueeze(1)
        kw = kw.repeat(1, num_views)
        kw = kw.reshape((-1,))
        kw = torch.ones_like(kw)
        kw.requires_grad = False

    def _unrolled_recon(self, coord, im, y, smaps, kw, denoiser):
        for u in range(self.cfg.unroll):
            # data consistency
            Ex = self.nufft_forward(coord, im, smaps)
            Exd = Ex - y
            # Exd = Exd * kw
            im = im - self.nufft_adjoint(coord, Exd, smaps) * self.scale_global

            # denoiser
            im = denoiser(im[None, None])
            im = im.squeeze()

            clear_gpu_mem()
        return im

    def _get_full_coords(self, kxc, kyc, kzc, readout, theta, radius_scale):
        """

        :param kxc, kyc, kzc: All 1D torch tensor of length Num_views. Coordinates of the helix centers
        :param readout: np.ndarray of shape (N_readout, 3)
        :param theta: 1D torch tensor
        :param radius_scale: 1D torch tensor
        :return: coord: (N_views * Nro, 3) tensor on cuda
        """
        num_views = kxc.shape[0]
        coords_centers = torch.stack((kxc, kyc, kzc), axis=1)
        readout_torch = torch.from_numpy(readout.copy()).cuda()


        coord = torch.zeros(readout.shape + (num_views,), dtype=torch.float32).cuda()
        for v in range(num_views):
            rot = torch.stack((torch.tensor(1., device='cuda'), torch.tensor(0., device='cuda'), torch.tensor(0., device='cuda'),
                               torch.tensor(0., device='cuda'), torch.cos(theta[v]), -torch.sin(theta[v]),
                               torch.tensor(0., device='cuda'), torch.sin(theta[v]), torch.cos(theta[v])))
            rot = torch.reshape(rot, (3, 3))
            scaler = torch.stack((torch.tensor(1., device='cuda'), radius_scale[v], radius_scale[v]))
            readout_tmp = readout_torch * scaler
            coord[..., v] = torch.matmul(readout_tmp, rot) + coords_centers[v].unsqueeze(0)
        coord = torch.permute(coord, (0, -1, 1))
        coord = torch.reshape(coord, (-1, NDIM))

        return coord


    def train_epoch(self, epoch: int) -> None:
        self.denoiser_bw.train()
        train_avg = RunningAverage()

        idx = np.arange(num_cases_train + num_cases_val)
        idx = np.delete(idx, [2, 14, 23, 35])
        idx = np.random.permutation(idx)

        # for each epoch, we go through all cases but only pick one random enc each case
        for cs in range(num_cases_train):
            self.optimizer.zero_grad()

            gaussian_level = random.uniform(self.cfg.gaussian_lower, self.cfg.gaussian_uppper)
            radius_scale_clamp = torch.tanh(self.radius_scale)
            kyc_clamp = self.activationC(self.kyc)
            kzc_clamp = self.activationC(self.kzc)

            # caseID<19 are FE enhanced scans (5pt encoding)
            # caseID>19 are non-contrast (4pt encoding)
            i = idx[cs]
            if i < 19:
                num_encs_case = num_enc_fe
            else:
                num_encs_case = num_enc_nc
            enc = np.random.randint(num_encs_case)
            imagesfile = os.path.join(self.cfg.data_folder_train, rf'{i}/Images.h5')
            smapsfile = os.path.join(self.cfg.data_folder_train, rf'{i}/SenseMaps.h5')

            truth = self._load_truth(imagesfile, enc)
            truth_noisy = add_gaussian_noise(truth, gaussian_level, percent=5)

            truth_tensor = torch.tensor(truth.copy(), dtype=torch.complex64)
            truth_tensor = truth_tensor.cuda()
            truth_tensor.requires_grad = False

            truth_tensor2 = torch.tensor(truth_noisy.copy(), dtype=torch.complex64)
            truth_tensor2 = truth_tensor2.cuda()
            truth_tensor2.requires_grad = False

            smaps = self._load_smaps(smapsfile)
            smaps_tensor = torch.from_numpy(smaps.copy()).type(torch.complex64)
            smaps_tensor.requires_grad = False

            coord = self._get_full_coords(self.kxc, kyc_clamp, kzc_clamp, self.theta, radius_scale_clamp)
            kw = self._get_kw(kyc_clamp.shape[0])

            # initial guess
            im = torch.zeros([XRES, YRES, ZRES], dtype=torch.complex64, device='cuda', requires_grad=True)

            # target
            y = self.nufft_forward(coord, truth_tensor2, smaps_tensor)

            # unrolled recon
            im = self._unrolled_recon(coord, im, y, smaps_tensor, kw, self.denoiser_bw)

            clear_gpu_mem()

            lossMSE = (im.squeeze() - truth_tensor.squeeze()).abs().pow(2).sum().pow(0.5) / truth_tensor.abs().pow(2).sum().pow(0.5)
            lossMSE.backward(retain_graph=True)
            self.optimizer.step()
            train_avg.update(lossMSE.detach().item())
            logging.info(f'Epoch {epoch} lossMSE =  {lossMSE.detach().item()}')

        self.writer_train.add_scalar('LossMSE', train_avg.avg(), epoch)
        self.writer_train.add_scalar('scale_global', self.scale_global.detach().cpu().numpy(), epoch)
        logging.info(f'scale_global={self.scale_global.detach().cpu().numpy()}')

        # save
        denoiserstate = {
                'state_dict': self.denoiser_bw.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'epoch': epoch
            }
        denoiserfile = os.path.join(self.cfg.log_dir, f'denoiser_{self.Ntrial}.pt')
        try:
            os.remove(denoiserfile)
        except OSError:
            pass
        torch.save(denoiserstate, denoiserfile)

        centers = np.stack((kyc_clamp.detach().cpu().numpy(), kzc_clamp.detach().cpu().numpy() ), axis=1)
        coordsfile = os.path.join(self.cfg.log_dir, f'centers_opt_3D_{self.Ntrial}.txt')
        try:
            os.remove(coordsfile)
        except OSError:
            pass
        np.savetxt(coordsfile, centers, fmt='%f', delimiter=",")

        scalerot = np.stack((radius_scale_clamp.detach().cpu().numpy(), self.theta.detach().cpu().numpy()), axis=1)
        scalerotfile = os.path.join(self.cfg.log_dir, f'scale_rot_{self.Ntrial}.txt')
        try:
            os.remove(scalerotfile)
        except OSError:
            pass
        np.savetxt(scalerotfile, scalerot, fmt='%f', delimiter=",")

        # plot sampling patterns on tensorboard and save training images
        if epoch % 5 == 0 or epoch == self.cfg.n_epochs - 1:
            rotplt, rotax = plt.subplots(figsize=(6, 6))
            rotscatter = rotax.scatter(kyc_clamp.detach().cpu().numpy() ,
                                       kzc_clamp.detach().cpu().numpy() ,
                                       c=self.theta.detach().cpu().numpy(), s=3, alpha=0.7)
            rotlegend = rotax.legend(*rotscatter.legend_elements(num=4),
                                     loc="upper right", title="Theta")
            rotax.add_artist(rotlegend)
            self.writer_train.add_figure('theta', rotplt, epoch)

            rotpltzoom, ax = plt.subplots(figsize=(6, 6))
            scatter = ax.scatter(kyc_clamp.detach().cpu().numpy(), kzc_clamp.detach().cpu().numpy(),
                                 c=self.theta.detach().cpu().numpy(), s=10, alpha=0.7, cmap='Spectral')
            plt.xlim(-15, 15)
            plt.ylim(-15, 15)
            legend = ax.legend(*scatter.legend_elements(num=4),
                               loc="upper right", title="Theta")
            ax.add_artist(legend)
            self.writer_train.add_figure('theta_zoom', rotpltzoom, epoch)

            scalerotplt, ax = plt.subplots(figsize=(6, 6))
            scatter = ax.scatter(kyc_clamp.detach().cpu().numpy(), kzc_clamp.detach().cpu().numpy() ,
                                 c=radius_scale_clamp.detach().cpu().numpy() * self.cfg.r, s=3, alpha=0.7)
            legend = ax.legend(*scatter.legend_elements(num=4),
                               loc="upper right", title="Radius")
            ax.add_artist(legend)
            self.writer_train.add_figure('radius', scalerotplt, epoch)

            scalerotpltzoom, ax = plt.subplots(figsize=(6, 6))
            scatter = ax.scatter(kyc_clamp.detach().cpu().numpy() , kzc_clamp.detach().cpu().numpy() ,
                                 c=radius_scale_clamp.detach().cpu().numpy() * self.cfg.r, s=10, alpha=0.7, cmap='Spectral')
            plt.xlim(-15, 15)
            plt.ylim(-15, 15)
            legend = ax.legend(*scatter.legend_elements(num=4),
                               loc="upper right", title="Radius")
            ax.add_artist(legend)
            self.writer_train.add_figure('radius_zoom', scalerotpltzoom, epoch)


        del im, y
        clear_gpu_mem()

    def validate_epoch(self, epoch: int) -> None:
        self.denoiser_bw.eval()
        eval_avg = RunningAverage()

        idx = [2,14,23,35]
        with torch.no_grad():
            for cs in range(num_cases_val):
                radius_scale_clamp = torch.tanh(self.radius_scale)
                kyc_clamp = self.activationC(self.kyc)
                kzc_clamp = self.activationC(self.kzc)

                i = idx[cs]
                if i < 19:
                    num_encs_case = num_enc_fe
                else:
                    num_encs_case = num_enc_nc
                enc = np.random.randint(num_encs_case)

                imagesfile = os.path.join(self.cfg.data_folder_train, rf'{i}/Images.h5')
                smapsfile = os.path.join(self.cfg.data_folder_train, rf'{i}/SenseMaps.h5')

                truth = self._load_truth(imagesfile, enc)
                truth_tensor = torch.tensor(truth.copy(), dtype=torch.complex64)
                truth_tensor = truth_tensor.cuda()
                truth_tensor.requires_grad = False

                smaps = self._load_smaps(smapsfile)
                smaps_tensor = torch.from_numpy(smaps.copy()).type(torch.complex64)
                smaps_tensor.requires_grad = False

                coord = self._get_full_coords(self.kxc, kyc_clamp, kzc_clamp, self.theta, radius_scale_clamp)
                kw = self._get_kw(kyc_clamp.shape[0])

                # initial guess
                im = torch.zeros([XRES, YRES, ZRES], dtype=torch.complex64, device='cuda', requires_grad=False)

                # target
                y = self.nufft_forward(coord, truth_tensor, smaps_tensor)

                # unrolled recon
                im = self._unrolled_recon(coord, im, y, smaps_tensor, kw, self.denoiser_bw)

                clear_gpu_mem()

                lossMSE_val = (im.squeeze() - truth_tensor.squeeze()).abs().pow(2).sum().pow(0.5) / truth_tensor.abs().pow(2).sum().pow(0.5)
                eval_avg.update(lossMSE_val.detach().item())

                # save images
                if epoch % 2 == 0 and i == 2:
                    with h5py.File(self.evalimagesfile, 'a') as hf:
                        if epoch == 0:
                            hf.create_dataset(f"eval_truth_mag", data=np.squeeze(np.abs(truth)))
                        hf.create_dataset(f"eval_im_epoch{epoch}_mag", data=np.squeeze(np.abs(im.detach().cpu().numpy())))


                del im, y
                torch.cuda.empty_cache()
            self.writer_val.add_scalar('LossMSE',  eval_avg.avg(), epoch)
            logging.info("Validation epoch %d", epoch)

    def run(self) -> None:

        for epoch in range(self.cfg.n_epochs):
            torch.cuda.empty_cache()

            self.train_epoch(epoch)
            self.validate_epoch(epoch)


def main() -> None:
    args = parse_args()
    Ntrial = randrange(10000)
    cfg = load_config(args.config)

    setup_logging(cfg.log_dir, Ntrial)
    setproctitle.setproctitle(f'optflow_{Ntrial}')
    logging.info(f'Setting program name to optflow_{Ntrial}')

    trainer = FlowTrainer(cfg, Ntrial)
    trainer.run()


if __name__ == "__main__":
    main()