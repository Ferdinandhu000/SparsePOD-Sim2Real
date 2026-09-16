import torch.nn.functional as F

def mse_loss(pred, target):
    return F.mse_loss(pred, target)
