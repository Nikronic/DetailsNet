# %% import library
from DetailsNet import DetailsNet
from Discriminators import DiscriminatorOne, DiscriminatorTwo
from torchvision.transforms import Compose, ToPILImage, ToTensor, RandomResizedCrop, RandomRotation, \
    RandomHorizontalFlip
from utils.preprocess import *
import torch
from torch.utils.data import DataLoader
from utils.loss import DetailsLoss

import torch.optim as optim
import torch.nn as nn
from torch.backends import cudnn

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# %% define data sets and their loaders
custom_transforms = Compose([
    RandomResizedCrop(size=224, scale=(0.8, 1.2)),
    RandomRotation(degrees=(-30, 30)),
    RandomHorizontalFlip(p=0.5),
    ToTensor(),
    RandomNoise(p=0.5, mean=0, std=1)])

train_dataset = PlacesDataset(txt_path='filelist.txt',
                              img_dir='data',
                              transform=custom_transforms)

train_loader = DataLoader(dataset=train_dataset,
                          batch_size=128,
                          shuffle=True,
                          num_workers=1,
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
        torch.nn.init.kaiming_normal_(m.weight, mode='fan_out')
        nn.init.constant_(m.bias, 0)
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
            y_descreen = random_noise_adder(y_descreen)
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


# %% test
def test_model(net, data_loader):
    """
    Return loss on test

    :param net: The trained NN network
    :param data_loader: Data loader containing test set
    :return: Print loss value over test set in console
    """

    net.eval()
    running_loss = 0.0
    with torch.no_grad():
        for data in data_loader:
            X = data['X']
            y_d = data['y_descreen']
            X = X.to(device)
            y_d = y_d.to(device)
            outputs = net(X)
            loss = criterion(outputs, y_d)
            running_loss += loss
            print('loss: %.3f' % running_loss)
    return outputs


def show_test(image_batch):
    """
    Get a batch of images of torch.Tensor type and show them as a single gridded PIL image

    :param image_batch: A Batch of torch.Tensor contain images
    :return: An array of PIL images
    """
    to_pil = ToPILImage()
    fs = []
    for i in range(len(image_batch)):
        img = to_pil(image_batch[i].cpu())
        fs.append(img)
    x, y = fs[0].size
    ncol = 3
    nrow = 3
    cvs = Image.new('RGB', (x * ncol, y * nrow))
    for i in range(len(fs)):
        px, py = x * int(i / nrow), y * (i % nrow)
        cvs.paste((fs[i]), (px, py))
    cvs.save('out.png', format='png')
    cvs.show()
    return fs


# %% run model
criterion = DetailsLoss()
random_noise_adder = RandomNoise(p=0, mean=0, std=1)
details_net = DetailsNet().to(device)
disc_one = DiscriminatorOne().to(device)
disc_two = DiscriminatorTwo().to(device)

optimizer = optim.Adam(details_net.parameters(), lr=0.0001)
# TODO add separate optimizer for each discriminator.

details_net.apply(init_weights)
disc_one.apply(init_weights)
disc_two.apply(init_weights)

train_model(details_net, train_loader, optimizer, criterion, discriminators=[disc_one, disc_two], epochs=10)

