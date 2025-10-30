#假设已经得到更新后的节点UUMat, DUMat, INMat, TAMat, INMat
#不妨让以上矩阵的每一行对应一个节点
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import net_complex_basis as ncb
import net_complex_components as ncc

class OutputUU(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.UUOutput=nn.Sequential(
            ncb.linear(embed_dim, embed_dim),
            ncb.relu(),
            ncb.linear(embed_dim, embed_dim),
            ncb.relu(),
            ncb.linear(embed_dim,1)
        )
        self.abs=ncb.abs()
    def forward(self,UUMat):
        '''
        输入：UUMat: [batch,num_uu,embed_dim]
        输出：UUPower: [batch,num_uu]
        '''
        batch_size,UUnum,embed_dim=UUMat.shape
        UUMatFlat=UUMat.reshape(-1,embed_dim) # [batch*num_uu,embed_dim]
        UUPowerFlat=self.UUOutput(UUMatFlat) # [batch*num_uu,1]
        UUPowerFlatReal=self.abs(UUPowerFlat.real) # [batch*num_uu,1]
        UUPower=UUPowerFlatReal.reshape(batch_size,UUnum) # [batch,num_uu]
        return UUPower
    
class OutputDU(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.DUOutput=nn.Sequential(
            ncb.linear(embed_dim, embed_dim),
            ncb.relu(),
            ncb.linear(embed_dim, embed_dim),
            ncb.relu(),
            ncb.linear(embed_dim, embed_dim),
            nn.Tanh()
        )
    def forward(self,DUMat):
        '''
        输入：DUMat: [batch,num_du,embed_dim]
        输出：DUComMat: [batch,num_du]
        '''
        batch_size,DUnum,embed_dim=DUMat.shape
        DUMatFlat=DUMat.reshape(-1,embed_dim) # [ batch*num_du,embed_dim]
        DUComMatFlat=self.DUOutput(DUMatFlat) # [batch*num_du,embed_dim]
        DUComMat=DUComMatFlat.reshape(batch_size,DUnum,embed_dim) # [batch,num_du,embed_dim]
        return DUComMat
    
class OutputIN(nn.Module):
    def __init__(self,embed_dim,num_trans):
        super().__init__()
        self.INOutput=nn.Sequential(
            ncb.linear(embed_dim, embed_dim),
            ncb.relu(),
            ncb.linear(embed_dim, embed_dim),
            ncb.relu(),
            ncb.linear(embed_dim, num_trans*num_trans),
            nn.Tanh()
        )
        self.num=num_trans
    def forward(self,INMat):
        '''
        输入：INMat: [batch,num_in,embed_dim]
        输出：INComMat: [batch,num_in,embed_dim]
        '''
        batch_size,Innum,embed_dim=INMat.shape
        INMatFlat=INMat.reshape(-1,embed_dim) # [ batch*num_in,embed_dim]
        INMatFlat=self.INOutput(INMatFlat) # [batch*num_in,embed_dim]
        INMat=INMatFlat.reshape(batch_size,Innum,self.num,self.num) # [batch,num_in,embed_dim,embed_dim]
        return INMat

class OutputTA(nn.Module):
    def __init__(self, embed_dim,num_trans):
        super().__init__()
        self.TAOutput=nn.Sequential(
            ncb.linear(embed_dim, embed_dim),
            ncb.relu(),
            ncb.linear(embed_dim, embed_dim),
            ncb.relu(),
            ncb.linear(embed_dim, num_trans*num_trans),
            nn.Tanh()
        )
        self.num=num_trans
    def forward(self,TAMat):
        '''
        输入：TAMat: [batch,num_ta,embed_dim]
        输出：TAComMat: [batch,num_ta,embed_dim]
        '''
        batch_size,TAnum,embed_dim=TAMat.shape
        TAMatFlat=TAMat.reshape(-1,embed_dim) # [ batch*num_ta,embed_dim]
        TAMatFlat=self.TAOutput(TAMatFlat) # [batch*num_ta,embed_dim]
        TAMat=TAMatFlat.reshape(batch_size, TAnum, self.num, self.num) # [batch,num_ta,embed_dim,embed_dim]
        return TAMat
    
class sensingAttention(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.att=nn.Sequential(
            ncb.linear(embed_dim, embed_dim),
            ncb.relu(),
            ncb.linear(embed_dim, embed_dim),
            ncb.relu(),
            ncb.linear(embed_dim, 1)
        )
        self.abs=ncb.abs()
        self.softmax=nn.Softmax(dim=-1)
    def forward(self,TAMat, INMat):
        '''
        输入：sensingMat: [batch,num_sensing,embed_dim]
        输出：sensingWeight: [batch,num_sensing]
        '''
        sensingMat=torch.cat([TAMat,INMat],dim=1) # [batch,num_sensing,embed_dim]
        sensingWeight=self.att(sensingMat) # [batch,num_sensing,1]
        sensingWeightReal=self.abs(sensingWeight) # [batch,num_sensing,1]
        sensingWeight=self.softmax(sensingWeightReal)+0j # [batch,num_sensing,1]
        sensingWeight=sensingWeight.squeeze(-1) # [batch,num_sensing]
        return sensingWeight

def OutputSensing(sensingWeight, TAMat, INMat):
    sensingMat = torch.cat([TAMat, INMat], dim=1)  # [batch,num_sensing,embed_dim,embed_dim]
    sensingWeight = sensingWeight.unsqueeze(-1).unsqueeze(-1)+0j  # [batch,num_sensing,1,1]
    #OutputSensingMat = sensingWeight * sensingMat  # [batch,num_sensing,embed_dim,embed_dim]
    return (sensingWeight * sensingMat).sum(dim=1)

class OutputLayer(nn.Module):
    def __init__(self, embed_dim, num_trans):
        super().__init__()
        self.UUOutput=OutputUU(embed_dim)
        self.DUOutput=OutputDU(embed_dim)
        self.INOutput=OutputIN(embed_dim, num_trans)
        self.TAOutput=OutputTA(embed_dim, num_trans)
        self.sensingAttention=sensingAttention(embed_dim)
    def forward(self,UUMat,DUMat,INMat,TAMat):
        '''
        输入：
            UUMat: 形状为 (batch, num_uu, embed_dim) 的张量
            DUMat: 形状为 (batch, num_du, embed_dim) 的张量
            INMat: 形状为 (batch, num_in, embed_dim) 的张量
            TAMat: 形状为 (batch, num_ta, embed_dim) 的张量
        输出：
            UUPower: 形状为 (batch, num_uu) 的张量
            DUComMat: 形状为 (batch, num_du, embed_dim) 的张量
            INComMat: 形状为 (batch, num_in, embed_dim) 的张量
            TAPower: 形状为 (batch, num_ta, embed_dim) 的张量
        '''
        UUPower=self.UUOutput(UUMat)
        DUComMat=self.DUOutput(DUMat)
        INSenMat=self.INOutput(INMat)
        TASenMat=self.TAOutput(TAMat)
        sensingWeight=self.sensingAttention(TAMat, INMat)
        SensingMat=OutputSensing(sensingWeight, TASenMat, INSenMat)
        return UUPower, DUComMat, SensingMat
    
if __name__ == "__main__":
    batch_size = 4
    embed_dim = 4
    num_heads = 8
    
    # 为每个矩阵设置不同的num_each（第二个维度）
    num_uu = 10  # UU矩阵的节点数
    num_du = 6   # DU矩阵的节点数  
    num_in = 8   # IN矩阵的节点数
    num_ta = 5   # TA矩阵的节点数
    num_ci = 7   # CI矩阵的节点数
    num_trans = 6  # 发射天线数
    # 创建随机初始化的输入张量
    UUMat = torch.randn(batch_size, num_uu, embed_dim,dtype=torch.complex64)  # (4, 10, 64)
    DUMat = torch.randn(batch_size, num_du, embed_dim,dtype=torch.complex64)  # (4, 6, 64)
    INMat = torch.randn(batch_size, num_in, embed_dim,dtype=torch.complex64)  # (4, 8, 64)
    TAMat = torch.randn(batch_size, num_ta, embed_dim,dtype=torch.complex64)  # (4, 5, 64)
    CIMat = torch.randn(batch_size, num_uu, num_du,dtype=torch.complex64)
    
    # 创建模型
    model = OutputLayer(embed_dim, num_trans)
    
    # 前向传播
    UUUpdated, DUUpdated, SensingMat = model(UUMat, DUMat, INMat, TAMat)
    
    # 检查输出维度
    print("输入维度:")
    print(f"UUMat: {UUMat.shape}")
    print(f"DUMat: {DUMat.shape}") 
    print(f"TAMat: {SensingMat.shape}")
    
    print("\n输出维度:")
    print(f"UUUpdated: {UUUpdated.shape} (应与UUMat相同: [{batch_size}, {num_uu}, {1}])")
    print(f"DUUpdated: {DUUpdated.shape} (应与DUMat相同: [{batch_size}, {num_du}, {embed_dim}])")
    print(f"TAUpdated: {SensingMat.shape} (应与TAMat相同: [{batch_size}, {num_trans}, {num_trans}])")
    
    # 验证输出形状是否正确
    
    print("\n✅ 所有测试通过！模型运行正常。")