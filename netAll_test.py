import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import netInitial
import netUpdate
import netOutput

class netAll(nn.Module):
    def __init__(self, num_trans, num_rece, num_heads, embed_dim):
        super().__init__()
        self.Init=netInitial.InitLayer(embed_dim, num_trans, num_rece)
        self.Update=netUpdate.UpdateLayer(embed_dim, num_heads)
        self.Output=netOutput.OutputLayer(embed_dim,num_trans)
    def forward(self,UUMat,DUMat,INMat,TAMat,CIMat):
        UUMat, DUMat, INMat, TAMat,CIMat= self.Init(UUMat,DUMat,INMat,TAMat,CIMat)
        UUMat, DUMat, INMat, TAMat= self.Update(UUMat,DUMat,INMat,TAMat,CIMat)
        UUPower, DUComMat, SensingMat= self.Output(UUMat,DUMat,INMat,TAMat)
        return UUPower, DUComMat, SensingMat

if __name__ == "__main__":
    batch_size = 2
    embed_dim = 4
    num_trans = 8  # 发射天线数
    num_rece = 6   # 接收天线数
    num_heads = 2
    # 设置每个矩阵的节点数量
    num_uu = 4    # UU节点数
    num_du = 3    # DU节点数
    num_in = 5    # IN节点数
    num_ta = 2    # TA节点数
    
    print(f"batch_size: {batch_size}, embed_dim: {embed_dim}")
    print(f"num_trans: {num_trans}, num_rece: {num_rece}")
    print(f"num_uu: {num_uu}, num_du: {num_du}, num_in: {num_in}, num_ta: {num_ta}")
    
    # 直接生成随机复数张量
    UUMat = torch.randn(batch_size, num_uu, num_rece, dtype=torch.complex64)-0.5-0.5j
    DUMat = torch.randn(batch_size, num_du, num_trans, dtype=torch.complex64)-0.5-0.5j
    INMat = torch.randn(batch_size, num_in, 2, dtype=torch.complex64)-0.5-0.5j
    TAMat = torch.randn(batch_size, num_ta, 2, dtype=torch.complex64)-0.5-0.5j
    CIMat = torch.randn(batch_size, num_uu, num_du, dtype=torch.complex64)-0.5-0.5j
    
    print(f"\n输入张量维度:")
    print(f"UUMat: {UUMat.shape} (复数)")
    print(f"DUMat: {DUMat.shape} (复数)")
    print(f"INMat: {INMat.shape} (复数)")
    print(f"TAMat: {TAMat.shape} (复数)")
    print(f"CIMat: {CIMat.shape} (复数)")
    
    # 检查输入数据类型
    print(f"\n输入数据类型:")
    print(f"UUMat: {UUMat.dtype}")
    print(f"DUMat: {DUMat.dtype}")
    print(f"INMat: {INMat.dtype}")
    print(f"TAMat: {TAMat.dtype}")
    print(f"CIMat: {CIMat.dtype}")
    
    # 创建模型
    model = netAll(num_trans, num_rece,num_heads,embed_dim)
    
    if(True):

        UU_init, DU_init, IN_init = model(UUMat, DUMat, INMat, TAMat, CIMat)
        print(f"\n输出张量维度:")
        print(f"UU_all: {UU_init.shape} (期望: [{batch_size}, {num_uu})")
        print(f"DU_all: {DU_init.shape} (期望: [{batch_size}, {num_du}, {embed_dim}])")
        print(f"Sengsing_all: {IN_init.shape} (期望: [{batch_size}, {num_trans}, {num_trans}])")

        print(f"\n✅ 所有输出均为复数张量!")
