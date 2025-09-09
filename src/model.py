import torch
import torch.nn as nn
import torch.nn.functional as F
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

class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = TrueComplexConv3d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.conv2 = TrueComplexConv3d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                TrueComplexConv3d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
            )
        self.activation = ComplexReLU()

    def forward(self, x):
        out = self.conv1(x)
        out = self.activation(out)
        out = self.conv2(out)
        out += self.shortcut(x)
        out = self.activation(out)
        return out

class ResNet(nn.Module):
    def __init__(self, block, num_blocks, in_planes=32):
        super(ResNet, self).__init__()
        self.in_planes = in_planes
        self.conv1 = nn.Conv2d(1, in_planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.layer1 = self._make_layer(block, in_planes, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, in_planes*2, num_blocks[1], stride=2)
        self.activation = ComplexReLU()

        self.conv_last = TrueComplexConv3d(in_planes*2, 1, kernel_size=3, stride=1, padding=1, bias=False)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def run_function(self):
        def custom_forward(x):
            out = self.conv1(x)
            out = self.activation(out)
            out = self.layer1(out)
            out = self.layer2(out)
            out = self.conv_last(out)
            torch.cuda.empty_cache()

            return out
        return custom_forward


    def forward(self, vol_input):
        out_final = checkpoint.checkpoint(self.run_function(), vol_input)

        return torch.squeeze(out_final)


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, fmaps=16, kernel_size=3, padding=1, stride=1, bias=False, true_complex=False):
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


    def forward(self, vol_input):
        out_final = checkpoint.checkpoint(self.run_function(), vol_input)
        return torch.squeeze(out_final) + torch.squeeze(vol_input)



class BlockWiseCNN(nn.Module):
    def __init__(self, denoiser, patch_size=None, overlap=None, use_reentrant=False):
        super(BlockWiseCNN, self).__init__()
        self.denoiser = denoiser
        self.patch_size = patch_size
        self.overlap = overlap
        self.use_reentrant = use_reentrant

    def _get_starts(self, dim, patch):
        if dim <= patch:
            return [0]
        starts = list(range(0, dim, patch))
        if starts[-1] != dim - patch:
            starts[-1] = dim - patch
        return starts

    def run_function(self, x):
        return self.denoiser(x, mode='nonblock')

    def forward(self, x):

        out = torch.zeros_like(x)

        patch_in = [p + o for p, o in zip(self.patch_size, self.overlap)]
        half_overlap = [o // 2 for o in self.overlap]


        # reflection pad the volume so that patches near the borders have enough context
        pad = (
            half_overlap[2], half_overlap[2],
            half_overlap[1], half_overlap[1],
            half_overlap[0], half_overlap[0],
        )
        x_pad = F.pad(x, pad=pad, mode="reflect")



        dims = x.shape[-3:]
        starts = [self._get_starts(d, p) for d, p in zip(dims, self.patch_size)]

        for z in starts[0]:
            for y in starts[1]:
                for x0 in starts[2]:
                    patch = x_pad[
                            ...,
                            z: z + patch_in[0],
                            y: y + patch_in[1],
                            x0: x0 + patch_in[2],
                            ]
                    patch = checkpoint.checkpoint(
                        self.run_function, patch, use_reentrant=self.use_reentrant
                    )
                    out[
                    ...,
                    z: z + self.patch_size[0],
                    y: y + self.patch_size[1],
                    x0: x0 + self.patch_size[2],
                    ] = patch[
                        ...,
                        half_overlap[0]: half_overlap[0] + self.patch_size[0],
                        half_overlap[1]: half_overlap[1] + self.patch_size[1],
                        half_overlap[2]: half_overlap[2] + self.patch_size[2],
                        ]

        return out