# %% libraries
import torch.nn as nn
import torch


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
        self.MSE_loss = nn.MSELoss(reduction='sum')
        self.BCE_loss = nn.BCELoss(reduction='mean')

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

    def forward(self, y, y_pred):
        # TODO y_pred and y are concatenated latent vector, so first we must extract different features.
        coarse_loss = 50 * self.l1_loss(y, y_pred) + 1 * self.MSE_loss(y, y_pred)
        edge_loss = self.BCE_loss(y, y_pred)
        raise NotImplementedError('Loss function of DetailsNet and Discriminators')
        patch_loss = None
        adversarial_loss = None

        loss = self.w1 * coarse_loss + self.w2 * edge_loss + self.w3 * patch_loss + self.w4 * adversarial_loss
        return loss
