import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
import numpy as np

class RunningAverage:
    def __init__(self):  # initialization
        self.count = 0
        self.sum = 0

    def reset(self):
        self.count = 0
        self.sum = 0

    def update(self, value, n=1):
        self.count += n
        self.sum += value * n

    def avg(self):
        return self.sum / self.count

class ComplexConv3d(nn.Module):
    """
        do conv on real and imag separately
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0,
                 dilation=1, groups=1, bias=False):
        super(ComplexConv3d, self).__init__()
        self.conv_r = nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)

    def forward(self, input):
        return self.conv_r(input.real) + 1j*self.conv_r(input.imag)

class TrueComplexConv3d(nn.Module):
    """
        do conv on real and imag separately
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0,
                 dilation=1, groups=1, bias=False):
        super(TrueComplexConv3d, self).__init__()
        self.conv_x = nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.conv_y = nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)

    def forward(self, input):
        re = self.conv_x(input.real) - self.conv_y(input.imag)
        im = self.conv_x(input.imag) + self.conv_y(input.real)
        return re + 1j * im

class ComplexReLU(nn.Module):
    '''
    A PyTorch module to apply relu activation on the magnitude of the signal. Phase is preserved
    '''
    def __init__(self):
        super(ComplexReLU, self).__init__()
        self.act = nn.ReLU(inplace=False)

    def forward(self, input):
        return self.act(input.real) + 1j*self.act(input.imag)


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, fmaps=32, kernel_size=3, padding=1, stride=1, bias=False, true_complex=False):
        super(ResBlock, self).__init__()
        if true_complex:
            self.conv1 = TrueComplexConv3d(in_channels, fmaps, kernel_size, padding=padding, bias=bias)
            self.conv2 = TrueComplexConv3d(fmaps, fmaps*2, kernel_size, padding=padding, stride=stride, bias=bias)
            self.conv3 = TrueComplexConv3d(fmaps*2, fmaps*2, kernel_size,padding=padding, bias=bias)
            self.conv4 = TrueComplexConv3d(fmaps*2, fmaps*2, kernel_size, padding=padding, bias=bias)
            self.conv8 = TrueComplexConv3d(fmaps*2, out_channels, kernel_size, padding=padding, bias=bias)

        else:
            self.conv0 = ComplexConv3d(in_channels, fmaps, 7, padding='same', bias=bias)

            self.conv1 = ComplexConv3d(in_channels, fmaps, kernel_size, padding=padding, bias=bias)
            self.conv2 = ComplexConv3d(fmaps, fmaps, kernel_size, padding=padding, stride=stride, bias=bias)
            # self.conv3 = ComplexConv3d(fmaps, fmaps, kernel_size,padding=padding, bias=bias)
            # self.conv4 = ComplexConv3d(fmaps, fmaps, kernel_size, padding=padding, bias=bias)
            self.conv8 = ComplexConv3d(fmaps, out_channels, kernel_size, padding=padding, bias=bias)

        self.activation = ComplexReLU()


    def run_function(self):
        def custom_forward(L):

            x = self.conv1(L)
            x = self.activation(x)
            x = self.conv2(x)
            x = self.activation(x)
            x = self.conv3(x)
            x = self.activation(x)
            x = self.conv4(x)
            x = self.activation(x)
            x = self.conv8(x)

            torch.cuda.empty_cache()

            return x

        return custom_forward

    def forward_nonblock(self, vol_input):
        out_final = checkpoint.checkpoint(self.run_function(), vol_input)
        return torch.squeeze(out_final) + torch.squeeze(vol_input)

    def forward(self, x, edge=20, blocks_per_dim=2, mode='nonblock'):
        return self.forward_nonblock(x)



class BlockWiseCNN(nn.Module):
    def __init__(self, denoiser, patch_size=None, overlap=None, use_reentrant=False):
        super(BlockWiseCNN, self).__init__()
        self.denoiser = denoiser
        self.patch_size = patch_size
        self.overlap = overlap
        self.use_reentrant = use_reentrant

    def run_function(self, x):
        return self.denoiser(x, mode='nonblock')

    def forward(self, x):
        # Size of patch fed to network
        N = [i + j for i, j in zip(self.patch_size, self.overlap)]

        # Overlap
        offset = [0,0,0]

        # Ns is the size of the output patch
        Ns = [i + j for i, j in zip(self.patch_size, offset)]

        # Patch into 8 overlapping blocks, the blocks are larger than the patch size
        # such that they have enough overlap to cover the receptive field of the network.
        # The patches are checkpointed to save memory and stiched back together.
        out = torch.zeros_like(x)

        patch0 = checkpoint.checkpoint(self.run_function, x[..., 0:N[0], 0:N[1], 0:N[2]], use_reentrant=self.use_reentrant)
        out[..., 0:Ns[0], 0:Ns[1], 0:Ns[2]] = patch0[..., 0:Ns[0], 0:Ns[1], 0:Ns[2]]

        patch1 = checkpoint.checkpoint(self.run_function, x[..., 0:N[0], 0:N[1], -N[2]:], use_reentrant=self.use_reentrant)
        out[..., 0:Ns[0], 0:Ns[1], -Ns[2]:] = patch1[..., 0:Ns[0], 0:Ns[1], -Ns[2]:]

        patch2 = checkpoint.checkpoint(self.run_function, x[..., 0:N[0], -N[1]:, 0:N[2]], use_reentrant=self.use_reentrant)
        out[..., 0:Ns[0], -Ns[1]:, 0:Ns[2]] = patch2[..., 0:Ns[0], -Ns[1]:, 0:Ns[2]]

        patch3 = checkpoint.checkpoint(self.run_function, x[..., 0:N[0], -N[1]:, -N[2]:], use_reentrant=self.use_reentrant)
        out[..., 0:Ns[0], -Ns[1]:, -Ns[2]:] = patch3[..., 0:Ns[0], -Ns[1]:, -Ns[2]:]

        patch4 = checkpoint.checkpoint(self.run_function, x[..., -N[0]:, 0:N[1], 0:N[2]], use_reentrant=self.use_reentrant)
        out[..., -Ns[0]:, 0:Ns[1], 0:Ns[2]] = patch4[..., -Ns[0]:, 0:Ns[1], 0:Ns[2]]

        patch5 = checkpoint.checkpoint(self.run_function, x[..., -N[0]:, 0:N[1], -N[2]:], use_reentrant=self.use_reentrant)
        out[..., -Ns[0]:, 0:Ns[1], -Ns[2]:] = patch5[..., -Ns[0]:, 0:Ns[1], -Ns[2]:]

        patch6 = checkpoint.checkpoint(self.run_function, x[..., -N[0]:, -N[1]:, 0:N[2]], use_reentrant=self.use_reentrant)
        out[..., -Ns[0]:, -Ns[1]:, 0:Ns[2]] = patch6[..., -Ns[0]:, -Ns[1]:, 0:Ns[2]]

        patch7 = checkpoint.checkpoint(self.run_function, x[..., -N[0]:, -N[1]:, -N[2]:], use_reentrant=self.use_reentrant)
        out[..., -Ns[0]:, -Ns[1]:, -Ns[2]:] = patch7[..., -Ns[0]:, -Ns[1]:, -Ns[2]:]

        return out
