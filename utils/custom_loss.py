import torch
from segmentation_models_pytorch.utils import base
from segmentation_models_pytorch.utils import functional as F
from segmentation_models_pytorch.base.modules import Activation


def tversky(pr, gt, eps=1e-7, alf=0.3, beta=0.7, 
            threshold=None, ignore_channels=None):
    """
    Calculate Tversky metric between predicted and ground truth masks
    Note: 
      - if alf = beta = 0.5, this reduces to Dice coefficent
      - if alf > beta, penalize false positives higher than false negatives
      - if alf < beta, penalize false negatives higher than false positives
    :param pr: predicted mask [torch.Tensor]
    :param gt: ground truth mask [torch.Tensor]
    :param alf: weight on false postive [float]
    :param bet: weight on false negative [float]
    :param eps: small constant to avoid zero division [float]
    :param threshold: threhsold for output binarization 
    :return : tversky metric [float]
    """
    
    pr = F._threshold(pr, threshold=threshold)
    pr, gt = F._take_channels(pr, gt, ignore_channels=ignore_channels)
    
    tp = torch.sum(gt * pr)
    fp = torch.sum(pr) - tp
    fn = torch.sum(gt) - tp
    
    score = (tp) / (tp + alf*fp + beta*fn + eps)
    
    return score
    

class TverskyLoss(base.Loss):
    
        def __init__(self, eps=1., alf=0.3, beta=0.7, activation=None, ignore_channels=None, **kwargs):
            super().__init__(**kwargs)
            self.eps = eps
            self.alf=alf
            self.beta = beta
            self.activation = Activation(activation)
            self.ignore_channels = ignore_channels
            
        def forward(self, y_pr, y_gt):
            y_pr = self.activation(y_pr)
            return 1 - tversky(
                y_pr, y_gt, 
                eps=self.eps,
                alf=self.alf,
                beta=self.beta, 
                threshold=None, 
                ignore_channels=self.ignore_channels,
            )