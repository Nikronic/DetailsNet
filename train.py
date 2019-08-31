# %% import library
from DetailsNet import DetailsNet
from Discriminators import DiscriminatorOne, DiscriminatorTwo
from torchvision.transforms import Compose, ToPILImage, ToTensor, RandomResizedCrop, RandomRotation, \
    RandomHorizontalFlip, Normalize
import torchvision.utils as vutils
from utils.preprocess import *
import torch
from torch.utils.data import DataLoader
from utils.loss import DetailsLoss

from torch.autograd import Variable
import torch.optim as optim
import torch.nn as nn
from torch.backends import cudnn
import argparse

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
Tensor = torch.cuda.FloatTensor if torch.cuda.is_available else torch.FloatTensor

# %% argparse

parser = argparse.ArgumentParser()
parser.add_argument("--txt", help='path to the text file', default='filelist.txt')
parser.add_argument("--img", help='path to the images tar(bug!) archive (uncompressed) or folder', default='data')
parser.add_argument("--txt_t", help='path to the text file of test set', default='filelist.txt')
parser.add_argument("--img_t", help='path to the images tar archive (uncompressed) of testset ', default='data')
parser.add_argument("--bs", help='int number as batch size', default=128, type=int)
parser.add_argument("--es", help='int number as number of epochs', default=10, type=int)
parser.add_argument("--nw", help='number of workers (1 to 8 recommended)', default=4, type=int)
parser.add_argument("--lr", help='learning rate of optimizer (=0.0001)', default=0.0001, type=float)
parser.add_argument("--cudnn", help='enable(1) cudnn.benchmark or not(0)', default=0, type=int)
parser.add_argument("--pm", help='enable(1) pin_memory or not(0)', default=0, type=int)
args = parser.parse_args()

if args.cudnn == 1:
    cudnn.benchmark = True
else:
    cudnn.benchmark = False

if args.pm == 1:
    pin_memory = True
else:
    pin_memory = False

# %% define data sets and their loaders
custom_transforms = Compose([
    RandomResizedCrop(size=224, scale=(0.8, 1.2)),
    RandomRotation(degrees=(-30, 30)),
    RandomHorizontalFlip(p=0.5),
    ToTensor(),
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    RandomNoise(p=0.5, mean=0, std=0.1)])

train_dataset = PlacesDataset(txt_path=args.txt,
                              img_dir=args.img,
                              transform=custom_transforms)

train_loader = DataLoader(dataset=train_dataset,
                          batch_size=args.bs,
                          shuffle=True,
                          num_workers=args.nw,
                          pin_memory=pin_memory)

test_dataset = PlacesDataset(txt_path=args.txt_t,
                             img_dir=args.img_t,
                             transform=ToTensor(),
                             test=True)

test_loader = DataLoader(dataset=test_dataset,
                         batch_size=args.bs,
                         shuffle=False,
                         num_workers=args.nw,
                         pin_memory=pin_memory)


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
            gt = data['y_descreen']
            noise = data['y_noise']

            gt = gt.to(device)
            noise = noise.to(device)
            noise = random_noise_adder(noise)

            # train discriminators
            disc_one_optim.zero_grad()
            disc_two_optim.zero_grad()

            # Disc one
            Ia = 0  # output of coarse_net
            ground_truth_residual = gt - Ia
            disc_one_out = disc_one(ground_truth_residual)
            valid = torch.ones(disc_one_out.size()).to(device)
            real_loss_d1 = criterion(disc_one_out, valid)
            real_loss_d1.backward()

            # Disc Two
            object_output = torch.Tensor().to(device)
            disc_two_out = disc_two(torch.cat((noise, object_output), dim=1))  # TODO check concatenated latent vector
            valid = torch.ones(disc_two_out.size()).to(device)
            real_loss_d2 = criterion(disc_two_out, valid)
            real_loss_d2.backward()

            # fake image
            gen_imgs = details_net(noise)

            # Disc one
            disc_one_out = disc_one(gen_imgs)
            fake = torch.zeros(disc_one_out.size()).to(device)
            fake_loss_d1 = criterion(disc_one_out, fake)
            fake_loss_d1.backward()

            # Disc two
            disc_two_out = disc_two(torch.cat((gen_imgs.detach(), object_output), dim=1))
            fake = torch.zeros(disc_two_out.size()).to(device)
            fake_loss_d2 = criterion(disc_two_out, fake)
            fake_loss_d2.backward()

            # Disc one and two
            disc_one_optim.step()
            disc_two_optim.step()

            # train generator
            details_optim.zero_grad()

            disc_one_out = disc_one(gen_imgs.detach())
            valid = torch.ones(disc_one_out.size()).to(device)
            loss_g1 = criterion(disc_one_out, valid)

            disc_two_out = disc_two(gen_imgs.detach())
            valid = torch.ones(disc_two_out.size()).to(device)
            loss_g2 = criterion(disc_two_out, valid)

            loss_g = loss_g1 + loss_g2
            # loss_g.requires_grad = True
            loss_g.backward()
            details_optim.step()

            running_loss_g += loss_g.item()
            running_loss_disc_one += fake_loss_d1.item() + real_loss_d1.item()
            running_loss_disc_two += fake_loss_d2.item() + real_loss_d2.item()

            vutils.save_image(gen_imgs.cpu().data,
                              'fake_samples_epoch_%s.png' % (str(epoch) + "_" + str(i + 1)),
                              normalize=False)

            print('[%d, %5d] loss_g: %.3f , loss_d1: %0.f, loss_d2: %0.f' %
                  (epoch + 1, i + 1, running_loss_g, running_loss_disc_one, running_loss_disc_two))
    print('Finished Training')


# %% test
def test_model(net, data_loader, criterion):
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


def show_image_batch(image_batch, name='out.png'):
    """
    Get a batch of images of torch.Tensor type and show them as a single gridded PIL image
    
    :param image_batch: A Batch of torch.Tensor contain images
    :param name: Name of output image
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
    cvs.save(name, format='png')
    cvs.show()
    return fs


# %% run model
# details_crit = DetailsLoss()

# to simplify implementation for demonstration purposes, I just use MSE loss just like LSGAN
# Final and fully implemented model can be found here : https://github.com/Nikronic/Deep-Halftoning

random_noise_adder = RandomNoise(p=0, mean=0, std=0.1)
details_net = DetailsNet(input_channels=3).to(device)
disc_one = DiscriminatorOne().to(device)
disc_two = DiscriminatorTwo(input_channel=3).to(device)

details_optim = optim.Adam(details_net.parameters(), lr=args.lr)
disc_one_optim = optim.Adam(disc_one.parameters(), lr=args.lr)
disc_two_optim = optim.Adam(disc_two.parameters(), lr=args.lr)

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

train_model(models, train_loader, optimizer=optims, criterion=nn.MSELoss(), epochs=args.es)
