import torch
import torch.nn as nn
import torch.nn.functional as F

from config import TRIPLET_MARGIN

class TripletLoss(nn.Module):
    """
    Triplet loss function.
    L(A, P, N) = max(0, D(A, P) - D(A, N) + margin)
    """
    def __init__(self, margin=TRIPLET_MARGIN):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, dist_ap, dist_an):
        """
        Args:
            dist_ap: Euclidean distances between anchor and positive samples. Shape: [B]
            dist_an: Euclidean distances between anchor and negative samples. Shape: [B]
        Returns:
            loss: The triplet loss value.
        """
        losses = F.relu(dist_ap - dist_an + self.margin)
        loss = losses.mean() # Average loss over the batch
        return loss
