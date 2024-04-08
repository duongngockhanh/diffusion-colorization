import torch
import torch.nn as nn
import torch.nn.functional as F


def mse_loss(output, target):
    return F.mse_loss(output, target)


def mae(input, target):
    with torch.no_grad():
        loss = nn.L1Loss()
        output = loss(input, target)
    return output