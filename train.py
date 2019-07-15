# %% import library
from DetailsNet import DetailsNet
from Discriminators import DiscriminatorOne, DiscriminatorTwo
from torchvision.transforms import Compose, ToPILImage, ToTensor, RandomResizedCrop, RandomRotation, \
    RandomHorizontalFlip, Normalize
from utils.preprocess import *
import torch
from torch.utils.data import DataLoader
from utils.loss import DetailsLoss

from torch.autograd import Variable
import torch.optim as optim
import torch.nn as nn
from torch.backends import cudnn

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
Tensor = torch.cuda.FloatTensor if torch.cuda.is_available else torch.FloatTensor

# %% define data sets and their loaders
custom_transforms = Compose([
    RandomResizedCrop(size=224, scale=(0.8, 1.2)),
    RandomRotation(degrees=(-30, 30)),
    RandomHorizontalFlip(p=0.5),
    ToTensor(),
    Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    RandomNoise(p=0.5, mean=0, std=0.0007)])

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
        torch.nn.init.kaiming_normal_(m.weight.data, mode='fan_out')
        nn.init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.BatchNorm2d):  # reference: https://github.com/pytorch/pytorch/issues/12259
        nn.init.constant_(m.weight.data, 1)
        nn.init.constant_(m.bias.data, 0)


# %% train model
def train_model(network, data_loader, optimizer, criterion, epochs=10):
    """
    Train model

    :param network: Parameters of defined neural network consisting of a generator and two discriminators
    :param data_loader: A data loader object defined on train data set
    :param epochs: Number of epochs to train model
    :param optimizer: Optimizer(s) to train network
    :param criterion: The loss function to minimize by optimizer
    :return: None
    """

    details_net = network['details'].train()
    disc_one = network['disc1'].train()
    disc_two = network['disc2'].train()

    # details_crit = criterion['details']
    # disc_one_crit = criterion['disc1']
    # disc_two_crit = criterion['disc2']

    details_optim = optimizer['details']
    disc_one_optim = optimizer['disc1']
    disc_two_optim = optimizer['disc2']

    for epoch in range(epochs):

        running_loss_g = 0.0
        running_loss_disc_one = 0.0
        running_loss_disc_two = 0.0
        for i, data in enumerate(data_loader, 0):
            y_d = data['y_descreen']
            y_noise = data['y_noise']

            valid = torch.ones(y_d.size(0), 1).fill_(1.0)
            fake = torch.zeros(y_d.size(0), 1).fill_(0.0)

            y_d = y_d.to(device)
            y_d = random_noise_adder(y_d)
            y_noise = y_noise.to(device)

            # train generator
            details_optim.zero_grad()

            gen_imgs = details_net(y_noise)
            g_loss = criterion(disc_one(gen_imgs), valid)
            g_loss.backward()
            details_optim.step()

            # train discriminator one
            disc_one_optim.zero_grad()

            Ia = 0  # output of coarse_net
            ground_truth_residual = y_d - Ia
            real_loss = criterion(disc_one(ground_truth_residual), valid)
            fake_loss = criterion(disc_one(gen_imgs), fake)
            disc_one_loss = (real_loss + fake_loss) / 2
            disc_one_loss.backward()
            disc_one_optim.step()

            # train discriminator two
            disc_two_optim.zero_grad()

            object_output = torch.Tensor()
            real_loss = criterion(disc_two(torch.cat((y_d, object_output)), dim=1), valid)
            fake_loss = criterion(disc_two(torch.cat((gen_imgs, object_output)), dim=1), fake)
            disc_two_loss = (real_loss + fake_loss) / 2
            disc_two_loss.backward()
            disc_two_optim.step()

            running_loss_g += g_loss.item()
            running_loss_disc_one += disc_one_loss.item()
            running_loss_disc_two += disc_two_loss.item()

            print('[%d, %5d] loss_g: %.3f , loss_d1: %0.f, loss_d2: %0.f' %
                  (epoch + 1, i + 1, running_loss_g, running_loss_disc_one, running_loss_disc_two))
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


def show_image_batch(image_batch):
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
    ncol = int(np.ceil(np.sqrt(len(image_batch))))
    nrow = int(np.ceil(np.sqrt(len(image_batch))))
    cvs = Image.new('RGB', (x * ncol, y * nrow))
    for i in range(len(fs)):
        px, py = x * int(i / nrow), y * (i % nrow)
        cvs.paste((fs[i]), (px, py))
    cvs.save('out.png', format='png')
    cvs.show()
    return fs


# %% run model
# details_crit = DetailsLoss()

# to simplify implementation for demonstration purposes, I just use MSE loss just like LSGAN
# Final and fully implmeneted model can be found here : https://github.com/Nikronic/Deep-Halftoning

random_noise_adder = RandomNoise(p=0, mean=0, std=0.0007)
details_net = DetailsNet(input_channels=3).to(device)
disc_one = DiscriminatorOne().to(device)
disc_two = DiscriminatorTwo(input_channel=3).to(device)

details_optim = optim.Adam(details_net.parameters(), lr=0.0001)
disc_one_optim = optim.Adam(disc_one.parameters(), lr=0.0001)
disc_two_optim = optim.Adam(disc_two.parameters(), lr=0.0001)

details_net.apply(init_weights)
disc_one.apply(init_weights)
disc_two.apply(init_weights)

models = {
    'details': details_net,
    'disc1': disc_one,
    'disc2': disc_two
}

# losses = {
#     'details': details_crit,
#     'disc1': disc_one,
#     'disc2': disc_two
# }

optims = {
    'details': details_optim,
    'disc1': disc_one_optim,
    'disc2': disc_two_optim
}

# %% train model

train_model(models, train_loader, optimizer=optims, criterion=nn.MSELoss(), epochs=1)
