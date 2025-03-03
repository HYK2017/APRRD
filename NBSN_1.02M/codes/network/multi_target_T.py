import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .baseline_network import rotate, unrotate, super_shift

class Inf_T_elements(nn.Module):
    def __init__(self, Trained_ATBSN):
        super(Inf_T_elements, self).__init__()
        self.Trained_ATBSN = Trained_ATBSN
        self.rotate = rotate()
        self.shift = super_shift()
        self.unrotate = unrotate()
    
    def forward(self, x, shift_factors=[0, 1, 2, 3, 4, 5], padding=16):
        x = F.pad(x, (padding, padding, padding, padding), mode='reflect')
        
        x = self.rotate(x)
        x = self.Trained_ATBSN.unet(x)
        xs = [self.shift(x, 2 * shift_factors[factor] - 1) for factor in shift_factors]
        xs = [self.unrotate(x_) for x_ in xs]
        
        x0s = [F.leaky_relu_(self.Trained_ATBSN.nin_A(x_), negative_slope=0.1) for x_ in xs]
        x0s = [F.leaky_relu_(self.Trained_ATBSN.nin_B(x0_), negative_slope=0.1) for x0_ in x0s]
        x0s = [self.Trained_ATBSN.nin_C(x0_) for x0_ in x0s]
        
        x0s = [x0_[:,:, padding:-padding, padding:-padding] for x0_ in x0s]
        return x0s  # T_elements