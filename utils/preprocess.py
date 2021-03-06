from __future__ import print_function, division
from PIL import Image
from torchvision.transforms import ToTensor, ToPILImage, Compose, Normalize
import numpy as np

import random
import tarfile
import io
import os
import pandas as pd

import torch
from torch.utils.data import Dataset


# %% custom dataset
class PlacesDataset(Dataset):
    def __init__(self, txt_path='filelist.txt', img_dir='data', transform=None, test=False):
        """
        Initialize data set as a list of IDs corresponding to each item of data set

        :param img_dir: path to image files as a uncompressed tar archive
        :param txt_path: a text file containing names of all of images line by line
        :param transform: apply some transforms like cropping, rotating, etc on input image
        :param test: is inference time or not
        :return: a 3-value dict containing input image (y_descreen) as ground truth,
         input image X as halftone image and edge-map (y_edge) of ground truth image to feed into the network.
        """

        df = pd.read_csv(txt_path, sep=' ', index_col=0)
        self.img_names = df.index.values
        self.txt_path = txt_path
        self.img_dir = img_dir
        self.transform = transform
        self.to_tensor = ToTensor()
        self.to_pil = ToPILImage()
        self.get_image_selector = True if img_dir.__contains__('tar') else False
        self.tf = tarfile.open(self.img_dir) if self.get_image_selector else None
        self.transform_gt = transform if test else Compose(self.transform.transforms[:-1])  # omit noise of ground truth

    def get_image_from_tar(self, name):
        """
        Gets a image by a name gathered from file list csv file

        :param name: name of targeted image
        :return: a PIL image
        """
        # tarinfo = self.tf.getmember(name)
        image = self.tf.extractfile(name)
        image = image.read()
        image = Image.open(io.BytesIO(image))
        return image

    def get_image_from_folder(self, name):
        """
        gets a image by a name gathered from file list text file

        :param name: name of targeted image
        :return: a PIL image
        """

        image = Image.open(os.path.join(self.img_dir, name))
        return image

    def __len__(self):
        """
        Return the length of data set using list of IDs

        :return: number of samples in data set
        """
        return len(self.img_names)

    def __getitem__(self, index):
        """
        Generate one item of data set. Here we apply our preprocessing things like halftone styles and
        subtractive color process using CMYK color model, generating edge-maps, etc.

        :param index: index of item in IDs list

        :return: a sample of data as a dict
        """

        if index == (self.__len__() - 1) and self.get_image_selector:  # Close tarfile opened in __init__
            self.tf.close()

        if self.get_image_selector:  # note: we prefer to extract then process!
            y_descreen = self.get_image_from_tar(self.img_names[index])
        else:
            y_descreen = self.get_image_from_folder(self.img_names[index])

        seed = np.random.randint(2147483647)
        random.seed(seed)

        if self.transform is not None:
            y_noise = self.noisy_image(y_descreen)
            y_descreen = self.transform(y_descreen)
            random.seed(seed)
            y_noise = self.transform_gt(y_noise)

        sample = {'y_descreen': y_descreen,
                  'y_noise': y_noise}

        return sample

    def noisy_image(self, image):
        """
        Add Salt and Pepper noise to image and return image as same type as input.

        :param image: PIL image
        :return: PIL image
        """

        if type(image) == torch.Tensor:
            image = self.to_pil(image)
        image = np.array(image)
        s_vs_p = 0.5
        amount = 0.015
        out = np.copy(image)
        num_salt = np.ceil(amount * image.size * s_vs_p)
        coords = tuple([np.random.randint(0, i - 1, int(num_salt)) for i in image.shape])
        out[coords] = 1
        num_pepper = np.ceil(amount * image.size * (1. - s_vs_p))
        coords = tuple([np.random.randint(0, i - 1, int(num_pepper)) for i in image.shape])
        out[coords] = 0
        out = ToPILImage()(out)
        return out


class RandomNoise(object):
    def __init__(self, p, mean=0, std=0.1): 
        self.p = p
        self.mean = mean
        self.std = std

    def __call__(self, img):
        if random.random() <= self.p:
            noise = torch.empty(*img.size(), dtype=torch.float, requires_grad=False)
            return img+noise.normal_(self.mean, self.std)
        return img


class Blend(object):
    """
    Blend two input tensors(tensors) with respect to the alpha value as a weight if random number is lower than p
    for each example
    """

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, halftone, ground_truth, alpha=0.5):
        """

        :param halftone: First tensor to be blended (batch_size, channel_size, height, width)
        :param ground_truth: Second tensor to be blended with size (batch_size, channel_size, height, width)
        :param alpha: weight of linear addition of two tensors

        :return: A tensor with size of (batch_size, channel_size, height, width)
        """

        p = torch.zeros(halftone.size()[0]).new_full((halftone.size()[0], ), self.p)
        rand = torch.zeros(p.size()[0]).uniform_()
        blend = torch.zeros((halftone.size()))
        mask = rand < p
        blend[mask] = halftone[mask] * (1.0 - alpha) + ground_truth[mask] * alpha
        mask = rand > p
        blend[mask] = halftone[mask]
        return blend


class UnNormalizeNative(object):
    """
    Unnormalize an input tensor given the mean and std
    """

    def __init__(self, mean, std):
        self.mean = torch.tensor(mean)
        self.std = torch.tensor(std)

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """

        return Normalize((-mean / std).tolist(), (1.0 / std).tolist())