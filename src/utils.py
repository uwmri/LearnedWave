import numpy as np
import sigpy as sp
import cupy as cp
import torch
import logging
import math

def hann(x, width):
    xp = backend.get_array_module(x)
    return .5 * (1 - xp.cos(2*math.pi*(x-width/2)/width))

def pca_coil_compression(kdata=None, axis=0, target_channels=None):
    logger = logging.getLogger('PCA_CoilCompression')

    if isinstance(kdata, list):
        logger.info('Passed k-space is a list, using encode 0 for compression')
        kdata_cc = kdata[0]
    else:
        kdata_cc = kdata

    logger.info(f'Compressing to {target_channels} channels, along axis {axis}')
    logger.info(f'Initial  size = {kdata_cc.shape} ')

    # Put channel to first axis
    kdata_cc = np.moveaxis(kdata_cc, axis, -1)
    old_channels = kdata_cc.shape[-1]
    logger.info(f'Old channels =  {old_channels} ')

    # Subsample to reduce memory for SVD
    mask_shape = np.array(kdata_cc.shape)
    mask = np.random.choice([True, False], size=mask_shape[:-1], p=[0.05, 1 - 0.05])

    # Create a subsampled array
    kcc = np.zeros((old_channels, np.sum(mask)), dtype=kdata_cc.dtype)
    logger.info(f'Kcc Shape = {kcc.shape} ')
    for c in range(old_channels):
        ktemp = kdata_cc[..., c]
        kcc[c, :] = ktemp[mask]

    kdata_cc = np.moveaxis(kdata_cc, -1, axis)

    #  SVD decomposition
    logger.info(f'Working on SVD of {kcc.shape}')
    u, s, vh = np.linalg.svd(kcc, full_matrices=False)

    logger.info(f'S = {s}')

    if isinstance(kdata, list):
        logger.info('Passed k-space is a list, using encode 0 for compression')

        for e in range(len(kdata)):
            kdata[e] = np.moveaxis(kdata[e], axis, -1)
            kdata[e] = np.expand_dims(kdata[e], -1)
            logger.info(f'Shape = {kdata[e].shape}')
            kdata[e] = np.matmul(u, kdata[e])
            kdata[e] = np.squeeze(kdata[e], axis=-1)
            kdata[e] = kdata[e][..., :target_channels]
            kdata[e] = np.moveaxis(kdata[e], -1, axis)

        for ksp in kdata:
            logger.info(f'Final Shape {ksp.shape}')
    else:
        # Now iterate over and multiply by u
        kdata = np.moveaxis(kdata, axis, -1)
        kdata = np.expand_dims(kdata, -1)
        kdata = np.matmul(u, kdata)
        logger.info(f'Shape = {kdata.shape}')

        # Crop to target channels
        kdata = np.squeeze(kdata, axis=-1)
        kdata = kdata[..., :target_channels]

        # Put back
        kdata = np.moveaxis(kdata, -1, axis)
        logger.info(f'Final shape = {kdata.shape}')

    return kdata

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

