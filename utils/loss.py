# %% libraries
import torch.nn as nn


class DetailsLoss(nn.Module):
    def __init__(self):
        """
        Return MSE Loss with mean of all losses in each mini-batch
        """
        super(DetailsLoss, self).__init__()
        self.MSE_loss = nn.MSELoss(reduction='mean')

    def forward(self, y, y_pred):
        loss = self.MSE_loss(y, y_pred)
        return loss
