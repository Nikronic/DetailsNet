# %% libraries
import torch.nn as nn
import torch
from vgg import vgg19_bn


class DetailsLoss(nn.Module):
    def __init__(self, w1=100, w2=0.1, w3=0.5, w4=1):
        """

        Return weighted sum of CoarseNet, EdgeNet, DetailsNet and Adversarial losses averaged over
        all losses in each mini-batch.

        :param w1: Weight of CoarseNet loss
        :param w2: Weight of EdgeNet loss
        :param w3: Weight of Local Patch loss
        :param w4: Weight of Adversarial loss
        """

        super(DetailsLoss, self).__init__()
        self.w1 = w1
        self.w2 = w2
        self.w3 = w3
        self.w4 = w4
        self.l1_loss = nn.L1Loss(reduction='mean')
        self.MSE_loss = nn.MSELoss(reduction='mean')
        self.BCE_loss = nn.BCELoss(reduction='mean')
        self.vgg19_bn = vgg19_bn(pretrained=True)

    # reference: https://github.com/pytorch/tutorials/blob/master/advanced_source/neural_style_tutorial.py
    @staticmethod
    def gram_matrix(mat):
        """
        Return Gram matrix

        :param mat: A matrix  (a=batch size(=1), b=number of feature maps,
        (c,d)=dimensions of a f. map (N=c*d))

        :return: Normalized Gram matrix
        """
        a, b, c, d = mat.size()
        features = mat.view(a * b, c * d)
        gram = torch.mm(features, features.t())
        return gram.div(a * b * c * d)

    @staticmethod
    def patch_feature(self):
        raise NotImplementedError('This method has not been implemented yet.')

    def forward(self, y, y_pred):
        """

        :param y: Ground truth tensor
        :param y_pred: Estimated ground truth
        :return: A scalar number
        """

        # TODO y_pred and y are concatenated latent vector, so first we must extract different features.

        y_patch_pool2 = self.patch_feature(self.vgg16_bn(y[0]))
        y_patch_pool5 = self.patch_feature(self.vgg16_bn(y[1]))
        y_pred_patch_pool2 = self.patch_feature(self.vgg16_bn(y_pred[0]))
        y_pred_patch_pool5 = self.patch_feature(self.vgg16_bn(y_pred[1]))

        coarse_loss = 50 * self.l1_loss(y, y_pred) + 1 * self.MSE_loss(y, y_pred)
        edge_loss = self.BCE_loss(y, y_pred)
        patch_loss = (self.MSE_loss(self.gram_matrix(y_patch_pool2), self.gram_matrix(y_pred_patch_pool2)) +
                      self.MSE_loss(self.gram_matrix(y_patch_pool5), self.gram_matrix(y_pred_patch_pool5)))
        adversarial_loss = None

        loss = self.w1 * coarse_loss + self.w2 * edge_loss + self.w3 * patch_loss + self.w4 * adversarial_loss
        raise NotImplementedError('Adversarial loss not implemented')
        return loss
