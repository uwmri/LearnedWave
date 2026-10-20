# util funcs for np/cp arrays

import numpy as np
import sigpy as sp
import cupy as cp
import logging
import math

def hann(x, width):
    xp = backend.get_array_module(x)
    return .5 * (1 - xp.cos(2*math.pi*(x-width/2)/width))

def pca_coil_compression(kdata=None, axis=0, target_channels=None, sample_frac=0.05, seed=0):

    """Compress coils to their ``target_channels`` principal components.

    Args:
        kdata: array with coils along ``axis``, or a list of such arrays (one
            per encoding). For a list, the compression matrix is estimated from
            entry 0 and applied to every entry, so all encodings share virtual
            coils.
        target_channels: channels to keep (all, reordered by energy, if None).
        sample_frac: fraction of samples used to estimate the coil covariance.

    Returns:
        The same structure with ``target_channels`` along ``axis``.
    """

    logger = logging.getLogger('PCA_CoilCompression')

    if isinstance(kdata, list):
        logger.info('Passed k-space is a list, using encode 0 for compression')

    ref = kdata[0] if isinstance(kdata, list) else kdata    # ref (C, ...)
    ref = np.moveaxis(ref, axis, -1)                        # ref (..., C)
    num_coils = ref.shape[-1]

    # Subsample to reduce memory for SVD
    rng = np.random.default_rng(seed)
    mask = rng.random(ref.shape[:-1]) < sample_frac         

    kcc = ref[mask].T  # (C, # of true values, flattened)
    logger.info(f"PCA coil compression {num_coils} -> {target_channels} "
                f"from {kcc.shape[1]} samples")
    u, s, _ = np.linalg.svd(kcc, full_matrices=False)       # u (C, C), s (C,), Vh (C, n_samples)
    logger.info(f"singular values: {s}")
    proj = u.conj().T[:target_channels]  # (target, C)

    def apply(k):
        k = np.moveaxis(k, axis, -1)                        # k (..., C)
        k = k @ proj.T                                      # k (..., target_channels)
        return np.moveaxis(k, -1, axis)                     # k with target_channels along original axis

    if isinstance(kdata, list):
        return [apply(k) for k in kdata]
    return apply(kdata)



def add_gaussian_noise(image,  level, percent=5):
    """
    Add Gaussian noise (sigma*level) in kspace, sigma is estimated from outer shell of the kspace.
    :param image: np or xp array of shape (h,w,d)
    :param level:
    :param percent: outer N% of the kspace
    :return: noisy image
    """
    device = sp.backend.get_device(image)
    xp = sp.backend.get_array_module(image)
    xres, yres, zres = image.shape()

    # get outer 5% shell
    sphere_center = (xres // 2, yres // 2, zres // 2)
    x, y, z = np.ogrid[:xres, :yres, :zres]
    distance = np.sqrt((x - sphere_center[0]) ** 2 + (y - sphere_center[1]) ** 2 + (z - sphere_center[2]) ** 2)
    sphere = distance <= xres//2
    ksp_central = np.percentile(distance[sphere], 1-percent)
    shell = sphere & (distance >= ksp_central)
    shell = sp.to_device(shell, device=device)
    ksp = sp.fourier.fft(image,center=True)

    # use the edge to estimate a realistic sigma
    kedge = ksp * shell
    kedge = kedge.reshape((-1,))
    kedge = kedge[np.nonzero(kedge)]
    sigma_est_real = np.median(np.abs(np.real(kedge)))      # or use mean
    sigma_est_imag = np.median(np.abs(np.imag(kedge)))
    sigma_real = level * sigma_est_real
    sigma_imag = level * sigma_est_imag
    
    gauss_real = xp.random.normal(0.0, sigma_real, (xres, yres, zres))
    gauss_imag = xp.random.normal(0.0, sigma_imag, (xres, yres, zres))
    knoisy_real = xp.real(ksp) + gauss_real
    knoisy_imag = xp.imag(ksp) + gauss_imag
    knoisy = knoisy_real + 1j * knoisy_imag
    image_noisy = sp.fourier.ifft(knoisy, center=True)

    return image_noisy


def clear_gpu_mem():
    cp._default_memory_pool.free_all_blocks()
    cp.get_default_pinned_memory_pool().free_all_blocks()
    torch.cuda.empty_cache()

