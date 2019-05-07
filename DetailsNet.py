import torch
import torch.nn as nn
import torch.nn.functional as F


class CL(nn.Module):
    def __init__(self, input_channel, output_channel, kernel_size=4, stride=2, padding=1):
        """
        It consists of the 4x4 convolutions with stride=2, padding=1, each followed by
        a leaky rectified linear unit (Leaky ReLU)

        :param input_channel: input channel size
        :param output_channel: output channel size
        """

        assert (input_channel > 0 and output_channel > 0)

        super(CL, self).__init__()
        layers = [nn.Conv2d(input_channel, output_channel, kernel_size=kernel_size, stride=stride, padding=padding),
                  nn.LeakyReLU(0.2)]
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


# %%
class CBL(nn.Module):
    def __init__(self, input_channel, output_channel, kernel_size=4, stride=2, padding=1):
        """
        It consists of the 4x4 convolutions with stride=2, padding=1, and a batch normalization, followed by
        a leaky rectified linear unit (ReLU)

        :param input_channel: input channel size
        :param output_channel: output channel size
        """
        assert (input_channel > 0 and output_channel > 0)

        super(CBL, self).__init__()
        layers = [nn.Conv2d(input_channel, output_channel, kernel_size=kernel_size, stride=stride, padding=padding),
                  nn.BatchNorm2d(num_features=output_channel), nn.LeakyReLU(0.2)]
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class C(nn.Module):
    def __init__(self, input_channel, output_channel):
        """
        At the final layer, a 3x3 convolution is used to map each 64-component feature vector to the desired
        number of classes.

        :param input_channel: input channel size
        :param output_channel: output channel size
        """
        super(C, self).__init__()
        layers = [nn.Conv2d(input_channel, output_channel, kernel_size=3, padding=1, stride=1),
                  nn.Sigmoid()]
        self.layer = nn.Sequential(*layers)

    def forward(self, x):
        return self.layer(x)


class ResidualBlock(nn.Module):
    def __init__(self, input_channel, output_channel, kernel_size=3, stride=1, padding=1):
        """
        A residual block contains a sequence of CBL and CL classes with same size of stride, padding and kernel size.

        :param input_channel: number of input channels of input images to network.
        :param output_channel: number of output channels of output images of network.
        :param stride: stride size of CBL and CL modules
        :param kernel_size: kernel_size of CBL and CL modules
        :param padding: padding size of CBL and CL modules
        """

        super(ResidualBlock, self).__init__()
        self.input_channel = input_channel
        self.output_channel = output_channel
        self.stride = stride
        self.kernel_size = kernel_size
        self.padding = padding

        # blocks
        self.cbl = CBL(input_channel, output_channel, kernel_size, stride, padding)
        self.cl = CL(input_channel, output_channel, kernel_size, stride, padding)

    def forward(self, x):
        out = self.cbl(x)
        out = self.cl(out)
        return out


class DetailsNet(nn.Module):
    def __init__(self, input_channels=64, output_channels=3):
        """
        The generator of GAN networks contains repeated residual blocks and C block at the end.

        :param input_channels: number of input channels of input images to network.
        :param output_channels: number of output channels of output images of network.
        """

        super(DetailsNet, self).__init__()
        self.input_channels = input_channels
        self.output_channels = output_channels

        self.c = C(input_channel=64, output_channel=3)

    def forward(self, x):
        pass


# from tensorboardX import SummaryWriter
#
# dummy_input = (torch.zeros(1, 3, 256, 256),)
#
# with SummaryWriter() as w:
#     w.add_graph(EdgeNet(input_channels=3, output_channels=1), dummy_input, True)
