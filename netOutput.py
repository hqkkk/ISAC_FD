#假设已经得到更新后的节点UUMat, DUMat, INMat, TAMat, INMat
#不妨让以上矩阵的每一行对应一个节点
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

class OutputUU(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.UUOutput=nn.Sequential(
            nn.Linear(embed_dim*2,embed_dim*2),
            nn.ReLU(),
            nn.Linear(embed_dim*2,embed_dim*2),
            nn.ReLU(),
            nn.Linear(embed_dim*2,1),
            nn.ReLU()
        )
    def forward(self,UUMat):
        '''
        输入：UUMat: [batch,num_uu,embed_dim*2]
        输出：UUPower: [batch,num_uu]
        '''
        batch_size,UUnum,embed_dim=UUMat.shape
        UUMatFlat=UUMat.reshape(-1,embed_dim*2) # [batch*num_uu,embed_dim*2]
        UUPowerFlat=self.UUOutput(UUMatFlat) # [batch*num_uu,1]
        UUPower=UUPowerFlat.reshape(batch_size,UUnum) # [batch,num_uu]
        return UUPower
    
class OutputDU(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.DUOutput=nn.Sequential(
            nn.Linear(embed_dim*2,embed_dim*2),
            nn.ReLU(),
            nn.Linear(embed_dim*2,embed_dim*2),
            nn.ReLU(),
            nn.Linear(embed_dim*2,embed_dim*2),
            nn.Tanh()
        )
    def forward(self,DUMat):
        '''
        输入：DUMat: [batch,num_du,embed_dim*2]
        输出：DUComMat: [batch,num_du]
        '''
        batch_size,DUnum,embed_dim=DUMat.shape
        DUMatFlat=DUMat.reshape(-1,embed_dim*2) # [ batch*num_du,embed_dim*2]
        DUComMatFlat=self.DUOutput(DUMatFlat) # [batch*num_du,embed_dim*2]
        DUComMat=DUComMatFlat.reshape(batch_size,DUnum,embed_dim*2) # [batch,num_du,embed_dim*2]
        return DUComMat
    
class OutputIN(nn.Module):
    def __init__(self,embed_dim):
        super().__init__()
        self.INOutput=nn.Sequential(
            nn.Linear(embed_dim*2,embed_dim*2),
            nn.ReLU(),
            nn.Linear(embed_dim*2,embed_dim*2),
            nn.ReLU(),
            nn.Linear(embed_dim*2,embed_dim*2),
            nn.Sigmoid()
        )
    def forward(self,INMat):
        '''
        输入：INMat: [batch,num_in,embed_dim*2]
        输出：INComMat: [batch,num_in,embed_dim*2]
        '''
        batch_size,Innum,embed_dim=INMat.shape
        INMatFlat=INMat.reshape(-1,embed_dim*2) # [ batch*num_in,embed_dim*2]
        INMatFlat=self.INOutput(INMatFlat) # [batch*num_in,embed_dim*2]
        INMat=INMatFlat.reshape(batch_size,Innum,embed_dim*2) # [batch,num_in,embed_dim*2]
        return INMat

class OutputTA(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.TAOutput=nn.Sequential(
            nn.Linear(embed_dim*2,embed_dim*2),
            nn.ReLU(),
            nn.Linear(embed_dim*2,embed_dim*2),
            nn.ReLU(),
            nn.Linear(embed_dim*2,embed_dim*embed_dim*2),
            nn.Sigmoid()
        )
    def forward(self,TAMat):
        '''
        输入：TAMat: [batch,num_ta,embed_dim*2]
        输出：TAComMat: [batch,num_ta,embed_dim*2]
        '''
        batch_size,TAnum,embed_dim=TAMat.shape
        TAMatFlat=TAMat.reshape(-1,embed_dim*2) # [ batch*num_ta,embed_dim*2]
        TAMatFlat=self.TAOutput(TAMatFlat) # [batch*num_ta,embed_dim*2]
        TAMat=TAMatFlat.reshape(batch_size, TAnum, embed_dim, embed_dim*2) # [batch,num_ta,embed_dim*2]
        return TAMat
    
class sensingAttention(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.att=nn.Sequential(
            nn.Linear(embed_dim*2,1),
            nn.Softmax(dim=1)
        )
    def forward(self,TAMat, INMat):
        '''
        输入：sensingMat: [batch,num_sensing,embed_dim*2]
        输出：sensingWeight: [batch,num_sensing]
        '''
        sensingMat=torch.cat([TAMat,INMat],dim=-1) # [batch,num_sensing,embed_dim*2]
        sensingWeight=self.att(sensingMat) # [batch,num_sensing,1]
        sensingWeight=sensingWeight.squeeze(-1) # [batch,num_sensing]
        return sensingWeight

def OutputSensing(sensingWeight, TAMat, INMat):
    sensingMat = torch.cat([TAMat, INMat], dim=-1)  # [batch,num_sensing,embed_dim,embed_dim*2]
    sensingWeight = sensingWeight.unsqueeze(-1).unsqueeze(-1)  # [batch,num_sensing,1,1]
    OutputSensingMat = sensingWeight * sensingMat  # [batch,num_sensing,embed_dim,embed_dim*2]
    return OutputSensingMat

def real2complex(realcomplex_tensor):
    """
    输入:
        real_tensor: 
    输出:
        complex_tensor
    """
    realpart = realcomplex_tensor[..., :realcomplex_tensor.shape[-1] // 2]
    imagpart = realcomplex_tensor[..., realcomplex_tensor.shape[-1] // 2:]
    complex_tensor = torch.complex(realpart, imagpart)
    return complex_tensor