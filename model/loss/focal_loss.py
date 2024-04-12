import torch
from torch import nn
from torch.nn import CrossEntropyLoss

class FocalLoss(nn.Module):
    
    def __init__(self,
                 weight=None,
                 gamma=1.0):

        assert gamma >= 0
        super().__init__()

        self.gamma = gamma
        self.weight = weight
        self.cross_entropy = CrossEntropyLoss(weight=self.weight, reduction="none")

    def forward(self, src, tar):
        
        CE = self.cross_entropy.to(src)(src, tar)
        p = torch.exp(- CE)
        loss = (1 - p) ** self.gamma * CE
        return loss.mean()