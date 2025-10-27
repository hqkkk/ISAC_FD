import numpy as np
import pandas as pd
import torch
import torch.nn as nn

class InitailLayer(nn.Module):
    def __init__(self, embed_dim, num_trans, num_rece):
        self.UUInit = nn.Sequential(
            nn.Linear(num_rece*2, embed_dim*2),
            nn.ReLU(),
            nn.Linear(embed_dim*2, embed_dim*2),
            nn.ReLU()
        )
        self.DUInit = nn.Sequential(
            nn.Linear(num_trans*2, embed_dim*2),
            nn.ReLU(),
            nn.Linear(embed_dim*2, embed_dim*2),
            nn.ReLU()
        )
        self.INInit = nn.Sequential(
            nn.Linear(4, embed_dim*2),
            nn.ReLU(),
            nn.Linear(embed_dim*2, embed_dim*2),
            nn.ReLU()
        )
        self.TAInit = nn.Sequential(
            nn.Linear(4, embed_dim*2),
            nn.ReLU(),
            nn.Linear(embed_dim*2, embed_dim*2),
            nn.ReLU()
        )
    
    def complex_to_real(self, complex_tensor):
        '''
        将复数张量转换为实数张量
        输入：complex_tensor: 形状为 (..., D) 的复数张量
        输出：real_tensor: 形状为 (..., 2*D) 的实数张量，实部和虚部分别存储
        '''
        real_part = complex_tensor.real
        imag_part = complex_tensor.imag
        real_tensor = torch.cat([real_part, imag_part], dim=-1)
        return real_tensor
    
    def forward(self, UUMat, DUMat, INMat, TAMat):
        '''
        输入：
            UUMat: 形状为 (batch, num_uu, num_rece) 的复数张量
            DUMat: 形状为 (batch, num_du, num_trans) 的复数张量
            INMat: 形状为 (batch, num_in, 2) 的复数张量
            TAMat: 形状为 (batch, num_ta, 2) 的复数张量
        输出：
            UU_init: 形状为 (batch, num_uu, embed_dim*2) 的实数张量
            DU_init: 形状为 (batch, num_du, embed_dim*2) 的实数张量
            IN_init: 形状为 (batch, num_in, embed_dim*2) 的实数张量
            TA_init: 形状为 (batch, num_ta, embed_dim*2) 的实数张量
        '''
        UUMat_real = self.complex_to_real(UUMat)  # (batch, num_uu, 2*num_rece)
        DUMat_real = self.complex_to_real(DUMat)  # (batch, num_du, 2*num_trans)
        INMat_real = self.complex_to_real(INMat)  # (batch, num_in, 4)
        TAMat_real = self.complex_to_real(TAMat)  # (batch, num_ta, 4)
        UUMat = self.UUInit(UUMat_real)  # (batch, num_uu, embed_dim*2)
        DUMat = self.DUInit(DUMat_real)  # (batch, num_du, embed_dim*2)
        INMat = self.INInit(INMat_real)  # (batch, num_in, embed_dim*2)
        TAMat = self.TAInit(TAMat_real)  # (batch, num_ta, embed_dim*2)
        return UUMat, DUMat, INMat, TAMat