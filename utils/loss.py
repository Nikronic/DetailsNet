# %% libraries
import torch.nn as nn


class EdgeLoss(nn.Module):
    def __init__(self):
        """
        Return Binary Entropy Loss with mean of all losses in each mini-batch
        """
        super(EdgeLoss, self).__init__()
        self.MSE_loss = nn.MSELoss(reduction='mean')

    def forward(self, y, y_pred):
        loss = self.MSE_loss(y, y_pred)
        return loss
