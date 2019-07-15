# %% import library
import torch
import torch.nn as nn
import torch.nn.functional as F


# %% submodules
class CL(nn.Module):
    def __init__(self, input_channel, output_channel, kernel_size=4, stride=2, padding=1):
        """
        It consists of the 4x4 convolutions with stride=2, padding=1, each followed by
        a leaky rectified linear unit (Leaky ReLU)

        :param input_channel: input channel size
        :param output_channel: output channel size
        :param kernel_size: kernel size of module
        :param stride: stride of module
        :param padding: padding of module
        """

        assert (input_channel > 0 and output_channel > 0)

        super(CL, self).__init__()
        layers = [nn.Conv2d(input_channel, output_channel, kernel_size=kernel_size, stride=stride, padding=padding),
                  nn.LeakyReLU(0.2, inplace=True)]
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class CBL(nn.Module):
    def __init__(self, input_channel, output_channel, kernel_size=4, stride=2, padding=1):
        """
        It consists of the 4x4 convolutions with stride=2, padding=1, and a batch normalization, followed by
        a leaky rectified linear unit (ReLU)

        :param input_channel: input channel size
        :param output_channel: output channel size
        :param kernel_size: kernel size of module
        :param stride: stride of module
        :param padding: padding of module
        """
        assert (input_channel > 0 and output_channel > 0)

        super(CBL, self).__init__()
        layers = [nn.Conv2d(input_channel, output_channel, kernel_size=kernel_size, stride=stride, padding=padding),
                  nn.BatchNorm2d(num_features=output_channel), nn.LeakyReLU(0.2, inplace=True)]
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class C(nn.Module):
    def __init__(self, input_channel, output_channel, kernel_size=3, stride=1, padding=1):
        """
        At the final layer, a 3x3 convolution is used to map feature vector to the desired
        number of classes.

        :param input_channel: input channel size
        :param output_channel: output channel size
        :param kernel_size: kernel size of module
        :param stride: stride of module
        :param padding: padding of module
        """

        super(C, self).__init__()
        layers = [nn.Conv2d(input_channel, output_channel, kernel_size=kernel_size, stride=stride, padding=padding)]
        self.layer = nn.Sequential(*layers)

    def forward(self, x):
        return self.layer(x)


# %% discriminator one
class DiscriminatorOne(nn.Module):
    def __init__(self, input_channel=3, output_channel=1):
        """
        Consists of a CL module followed by repetitive CBL modules and finally a C class
        to match the final needed classes.

        :param input_channel: number of input channels of input images to network.
        :param output_channel: number of output channels of input images to network.
        """

        super(DiscriminatorOne, self).__init__()
        self.cl = CL(input_channel=input_channel, output_channel=128, kernel_size=4, stride=2, padding=1)
        self.cbl0 = CBL(input_channel=128, output_channel=256, kernel_size=4, stride=2, padding=1)
        self.cbl1 = CBL(input_channel=256, output_channel=512, kernel_size=4, stride=2, padding=1)
        self.cbl2 = CBL(input_channel=512, output_channel=1024, kernel_size=4, stride=2, padding=1)
        self.cbl3 = CBL(input_channel=1024, output_channel=2048, kernel_size=4, stride=2, padding=1)

        self.final = C(input_channel=2048, output_channel=output_channel, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x = self.cl(x)
        x = self.cbl0(x)
        x = self.cbl1(x)
        x = self.cbl2(x)
        x = self.cbl3(x)
        x = self.final(x)
        x = x.view(x.size(0), -1)
        return x


# %% discriminator two
class DiscriminatorTwo(nn.Module):
    def __init__(self, input_channel=9, output_channel=1):
        """
        Consists of a CL module followed by repetitive CBL modules and finally a C class
        to match the final needed classes.

        :param input_channel: number of input channels of input images to network which is concatenation of
        I<sub>h</sub>, I<sub>d</sub>, and I<sub>o</sub> RGB vectors.
        :param output_channel: number of output channels of input images to network.
        """

        super(DiscriminatorTwo, self).__init__()
        self.cl = CL(input_channel=input_channel, output_channel=128, kernel_size=5, stride=2, padding=0)
        self.cbl0 = CBL(input_channel=128, output_channel=256, kernel_size=5, stride=2, padding=0)
        self.cbl1 = CBL(input_channel=256, output_channel=512, kernel_size=5, stride=2, padding=0)
        self.cbl2 = CBL(input_channel=512, output_channel=1024, kernel_size=5, stride=2, padding=0)
        self.cbl3 = CBL(input_channel=1024, output_channel=2048, kernel_size=5, stride=2, padding=0)

        self.final = C(input_channel=2048, output_channel=output_channel, kernel_size=4, stride=1, padding=0)

    def forward(self, x):
        x = self.cl(x)
        x = self.cbl0(x)
        x = self.cbl1(x)
        x = self.cbl2(x)
        x = self.cbl3(x)
        x = self.final(x)
        x = x.view(x.size(0), -1)
        return x


# %% tests
# z = torch.randn(size=(1, 3, 256, 256))
# d1 = DiscriminatorOne()
# z = d1(z)
# z.size()
