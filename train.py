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
    RandomNoise(p=0.5, mean=0, std=0.1)])

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
    :param optimizer: Optimizer(s) to train network
    :param criterion: The loss function to minimize by optimizer
    :return: None
    """

    optimizer_g = optimizer[0]
    optimizer_d1 = optimizer[1]
    optimizer_d2 = optimizer[2]

    net.train()
    for epoch in range(epochs):

        running_loss_g = 0.0
        running_loss_d = 0.0
        for i, data in enumerate(data_loader, 0):
            X = data['y_descreen']
            y_noise = data['y_noise']

            valid = Variable(Tensor(X.size(0), 1).fill_(1.0), requires_grad=False)
            fake = Variable(Tensor(X.size(0), 1).fill_(0.0), requires_grad=False)

            X = X.to(device)
            X = random_noise_adder(X)
            y_noise = y_noise.to(device)

            # train generator
            optimizer_g.zero_grad()

            gen_imgs = net(y_noise)
            g_loss = criterion(disc_one(gen_imgs), valid)
            g_loss.backward()
            optimizer_g.step()

            # train discriminator
            optimizer_d1.zero_grad()
            optimizer_d2.zero_grad()

            real_loss = criterion(disc_one(X), valid)
            fake_loss = criterion(disc_one(gen_imgs), fake)
            d_loss = (real_loss + fake_loss) / 2
            d_loss.backward()
            optimizer_d1.step()
            optimizer_d2.step()

            running_loss_g += g_loss.item()
            running_loss_d += d_loss.item()

            print('[%d, %5d] loss_g: %.3f , loss_d: %0.f' % (epoch + 1, i + 1, running_loss_g, running_loss_d))
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
criterion = DetailsLoss()
random_noise_adder = RandomNoise(p=0, mean=0, std=1)
details_net = DetailsNet().to(device)
disc_one = DiscriminatorOne().to(device)
disc_two = DiscriminatorTwo().to(device)

optimizer_g = optim.Adam(details_net.parameters(), lr=0.0001)
optimizer_d1 = optim.Adam(disc_one.parameters(), lr=0.0001)
optimizer_d2 = optim.Adam(disc_two.parameters(), lr=0.0001)

details_net.apply(init_weights)
disc_one.apply(init_weights)
disc_two.apply(init_weights)

train_model(details_net, train_loader, optimizer=[optimizer_g, optimizer_d1, optimizer_d2],
            criterion=criterion, discriminators=[disc_one, disc_two], epochs=10)
