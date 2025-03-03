import torch
import torch.nn as nn
import math

class L_APR(nn.Module):
    def __init__(self):
        super(L_APR, self).__init__()
        self.l1_loss = nn.L1Loss()

    def forward(self, f_h1_y, h2_y, h1_fy, h2_fy, reg_factor=4):
        L_rec = self.l1_loss(f_h1_y, h2_y)
        L_reg = torch.mean(torch.abs(f_h1_y - h2_y - (h1_fy - h2_fy)))
        return L_rec + reg_factor*L_reg
    
class L_RD(nn.Module):
    def __init__(self):
        super(L_RD, self).__init__()
        self.l1_loss = nn.L1Loss()

    def forward(self, fD_y, T_RD_elements):
        L_rd = sum(self.l1_loss(fD_y, T_RD) for T_RD in T_RD_elements)
        return L_rd
    
class CustomCosineLR:
    def __init__(self, optimizer, total_iter, current_iter=0, eta_min=0):
        self.optimizer = optimizer
        self.total_iter = total_iter
        self.current_iter = current_iter
        self.eta_min = eta_min
        self.base_lrs = [group['lr'] for group in optimizer.param_groups]
        self.last_lrs = self.get_lr()

    def get_lr(self):
        cos_inner = math.pi * (self.current_iter / self.total_iter)
        return [self.eta_min + (base_lr - self.eta_min) * (1 + math.cos(cos_inner)) / 2 for base_lr in self.base_lrs]
    
    def get_last_lr(self):
        return self.last_lrs

    def step(self):
        self.current_iter += 1
        self.last_lrs = self.get_lr()
        for param_group, lr in zip(self.optimizer.param_groups, self.last_lrs):
            param_group['lr'] = lr