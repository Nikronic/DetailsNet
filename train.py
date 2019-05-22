# %% import library
from DetailsNet import DetailsNet
from Discriminators import DiscriminatorOne, DiscriminatorTwo
from torchvision.transforms import Compose, ToPILImage, ToTensor, RandomResizedCrop, RandomRotation, \
    RandomHorizontalFlip
from utils.preprocess import *
import torch
from torch.utils.data import DataLoader
from utils.loss import DetailsLoss

from vgg import vgg19_bn
import torch.optim as optim
import torch.nn as nn
from torch.backends import cudnn

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# %% define data sets and their loaders
custom_transforms = Compose([
    RandomResizedCrop(size=224, scale=(0.8, 1.2)),
    RandomRotation(degrees=(-30, 30)),
    RandomHorizontalFlip(p=0.5),
    ToTensor()])

train_dataset = PlacesDataset(txt_path='filelist.txt',
                              img_dir='data',
                              transform=custom_transforms)

train_loader = DataLoader(dataset=train_dataset,
                          batch_size=128,
                          shuffle=True,
                          num_workers=5,
                          pin_memory=False)

test_dataset = PlacesDataset(txt_path='filelist.txt',
                             img_dir='data',
                             transform=ToTensor())

test_loader = DataLoader(dataset=test_dataset,
                         batch_size=128,
                         shuffle=False,
                         num_workers=0,
                         pin_memory=False)


# %% initialize network, loss and optimizer
def init_weights(m):
    """
    Initialize weights of layers using Kaiming Normal (He et al.) as argument of "Apply" function of
    "nn.Module"
    :param m: Layer to initialize
    :return: None
    """

    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        torch.nn.init.kaiming_normal_(m.weight, mode='fan_in')
        m.bias.data.fill_(0.0)
    elif isinstance(m, nn.BatchNorm2d):  # reference: https://github.com/pytorch/pytorch/issues/12259
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)


# %% train model
def train_model(net, data_loader, optimizer, criterion, discriminators=None, epochs=10):
    """
    Train model
    :param net: Parameters of defined neural network
    :param discriminators: List of discriminators objects
    :param data_loader: A data loader object defined on train data set
    :param epochs: Number of epochs to train model
    :param optimizer: Optimizer to train network
    :param criterion: The loss function to minimize by optimizer
    :return: None
    """

    net.train()
    for epoch in range(epochs):

        running_loss = 0.0
        for i, data in enumerate(data_loader, 0):
            y_descreen = data['y_descreen']
            y_noise = data['y_noise']

            y_descreen = y_descreen.to(device)
            y_noise = y_noise.to(device)

            optimizer.zero_grad()

            outputs = net(y_descreen)
            loss = criterion(outputs, y_noise)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss))
            running_loss = 0.0
    print('Finished Training')


# %% run model
criterion = DetailsLoss()
details_net = DetailsNet().to(device)
vgg19_bn_net = vgg19_bn(pretrained=True)
disc_one = DiscriminatorOne().to(device)
disc_two = DiscriminatorTwo().to(device)
optimizer = optim.Adam(details_net.parameters(), lr=0.0001)
details_net.apply(init_weights)
train_model(details_net, train_loader, optimizer, criterion, discriminators=[disc_one, disc_two], epochs=10)
