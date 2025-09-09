"""
sigpy NUFFT as autograd functions
modified based on Alban Gossard's implementation
https://github.com/albangossard/Bindings-NUFFT-pytorch
and Guanhua Wang https://github.com/guanhuaw/Bjork/blob/main/sys_op.py

"""
import os
import sigpy as sp
from sigpy.pytorch import to_pytorch, from_pytorch
import torch
from sympy.codegen.ast import String
from torch import Tensor



os.environ["CUPY_CACHE_SAVE_CUDA_SOURCE"] = "1"
os.environ["CUPY_DUMP_CUDA_SOURCE_ON_ERROR"] = "1"


class NUFFT(torch.autograd.Function):
    @staticmethod
    def forward(ctx, coord, im, smaps):
        """

        :param coord: Tensor (total pts, ndim). need to be on gpu
        :param im: Tensor (xres, yres, zres). need to be on gpu
        :param smaps: Tensor (c, xres, yres, zres). can be on cpu
        :return: kdata, tensor of (c, total pts) on gpu
        """
        ctx.save_for_backward(coord, im, smaps)
        A = Nufft_op3D(smaps, coord)
        output = A.forward(im)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        print(f'grad_output shape {grad_output.shape}')
        coord, im, smaps = ctx.saved_tensors
        A = Nufft_op3D(smaps, coord)
        grad_input = A.backward_forward(im, grad_output)
        grad_input_im = A.adjoint(grad_output)
        return grad_input, grad_input_im, None


class NUFFTadjoint(torch.autograd.Function):
    @staticmethod
    def forward(ctx, coord, y, smaps):
        ctx.save_for_backward(coord, y, smaps)
        A = Nufft_op3D(smaps, coord)
        output = A.adjoint(y)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        print(f'grad_output shape {grad_output.shape}')
        coord, y, smaps = ctx.saved_tensors
        A = Nufft_op3D(smaps, coord)
        grad_input = A.backward_adjoint(y, grad_output)      # (klength, 2)
        grad_input_y = A.forward(grad_output)

        return grad_input, grad_input_y, None


class Nufft_op3D():
    def __init__(self, smaps: Tensor, coord: Tensor, type: String):
        self.torch_dtype = torch.float32
        self.torch_cpxdtype = torch.complex64
        self.type = type
        self.smaps = smaps
        self.coord = coord
        self.device = coord.device
        self.A = sp.mri.linop.Sense(from_pytorch(smaps), coord=from_pytorch(coord), weights=None, tseg=None,
                                 coil_batch_size=1, comm=None,
                                 transp_nufft=False)
        self.nx = smaps.shape[1]
        self.ny = smaps.shape[2]
        self.nz = smaps.shape[3]

        self.image_shape = smaps.shape[1:]
        self.ncoils = smaps.shape[0]

        if coord.requires_grad:
            self.xx = torch.arange(self.nx, device=self.device, dtype=self.torch_dtype) - self.nx / 2.
            self.xy = torch.arange(self.ny, device=self.device, dtype=self.torch_dtype) - self.ny / 2.
            self.xz = torch.arange(self.nz, device=self.device, dtype=self.torch_dtype) - self.nz / 2.
            self.XX, self.XY, self.XZ = torch.meshgrid(self.xx, self.xy, self.xz)

    def forward(self, im: Tensor) -> Tensor:

        y = to_pytorch(self.A * from_pytorch(im))
        y = y.type(self.torch_cpxdtype)
        return y

    def adjoint(self, y: Tensor)-> Tensor:
        im = to_pytorch(self.A.H * from_pytorch(y))
        im = im.type(self.torch_cpxdtype)
        return im

    def backward_forward(self, im, g):
        grad = torch.zeros(self.coord.shape, dtype=self.torch_dtype, device=self.device)
        vec_fx = torch.mul(self.XX, im)
        vec_fy = torch.mul(self.XY, im)
        vec_fz = torch.mul(self.XZ, im)

        # print(f'backward_forward: vec_fx shape {vec_fx.shape}')     # torch.Size([256, 256, 256])

        # CT 03/2023 implementation of Guanhua's paper. See original nufftbindings for Alban's method.
        # The results are the same.
        xrd = to_pytorch(self.A * from_pytorch(vec_fx))
        grad[:, 0] = ((torch.conj(xrd.mul_(0 - 1j)).mul_(g)).real * 2).sum(axis=0)
        xrd = to_pytorch(self.A * from_pytorch(vec_fy))
        grad[:, 1] = ((torch.conj(xrd.mul_(0 - 1j)).mul_(g)).real * 2).sum(axis=0)
        xrd = to_pytorch(self.A * from_pytorch(vec_fz))
        grad[:, 2] = ((torch.conj(xrd.mul_(0 - 1j)).mul_(g)).real * 2).sum(axis=0)

        return grad

    def backward_adjoint(self, y, g):
        grad = torch.zeros(self.coord.shape, dtype=self.torch_dtype, device=self.device)

        vecx_grad_output = torch.mul(self.XX, g)
        vecy_grad_output = torch.mul(self.XY, g)
        vecz_grad_output = torch.mul(self.XZ, g)

        # CT 03/2023 implementation of Guanhua's paper. See original nufftbindings for Alban's method
        tmp = to_pytorch(self.A * from_pytorch(vecx_grad_output))
        grad[:, 0] = ((tmp.mul_(torch.conj(y) * (0 - 1j))).real * 2).sum(axis=0)
        tmp = to_pytorch(self.A * from_pytorch(vecy_grad_output))
        grad[:, 1] = ((tmp.mul_(torch.conj(y) * (0 - 1j))).real * 2).sum(axis=0)
        tmp = to_pytorch(self.A * from_pytorch(vecz_grad_output))
        grad[:, 2] = ((tmp.mul_(torch.conj(y) * (0 - 1j))).real * 2).sum(axis=0)

        return grad
