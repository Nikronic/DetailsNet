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
    def __init__(self, input_channel, output_channel, kernel_size=3, stride=1, padding=1):
        """
        At the final layer, a 3x3 convolution is used to map feature vector to the desired
        number of classes.

        :param input_channel: input channel size
        :param output_channel: output channel size
        """

        super(C, self).__init__()
        layers = [nn.Conv2d(input_channel, output_channel, kernel_size=kernel_size, stride=stride, padding=padding),
                  nn.Sigmoid()]
        self.layer = nn.Sequential(*layers)

    def forward(self, x):
        return self.layer(x)


class DiscriminatorOne(nn.Module):
    def __init__(self, input_channel, output_channel=1):
        """
        Consists of a CL module followed by repetitive CBL modules and finally a C class
        to match the final needed classes.

        :param input_channel: number of input channels of input images to network.
        :param output_channel: number of output channels of input images to network.
        """

        super(DiscriminatorOne, self).__init__()
        # TODO input size is ARBITRARY for now and must be calculated exactly!

    def forward(self, x):
        pass

# TODO add documentation to submodules for kernel_Size, stride and padding arguments.
