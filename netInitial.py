import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import net_complex_basis as ncb
import net_complex_components as ncc

class InitLayer(nn.Module):
    def __init__(self, embed_dim, num_trans, num_rece):
        super().__init__()
        self.UUInit = ncc.DNN_3layer(num_rece, embed_dim, embed_dim)
        self.DUInit = ncc.DNN_3layer(num_trans, embed_dim, embed_dim)
        self.INInit = ncc.DNN_3layer(2, embed_dim, embed_dim)
        self.TAInit = ncc.DNN_3layer(2, embed_dim, embed_dim)
        self.CIInit = ncc.DNN_3layer(1, 5, 1)
    def forward(self, UUMat, DUMat, INMat, TAMat, CIMat):
        '''
        输入：
            UUMat: 形状为 (batch, num_uu, num_rece) 的复数张量
            DUMat: 形状为 (batch, num_du, num_trans) 的复数张量
            INMat: 形状为 (batch, num_in, 2) 的复数张量
            TAMat: 形状为 (batch, num_ta, 2) 的复数张量
            CIMat: 形状为 (batch, num_uu, num_du) 的复数张量
        输出：
            UU_init: 形状为 (batch, num_uu, embed_dim) 的张量
            DU_init: 形状为 (batch, num_du, embed_dim) 的张量
            IN_init: 形状为 (batch, num_in, embed_dim) 的张量
            TA_init: 形状为 (batch, num_ta, embed_dim) 的张量
            CI_init: 形状为 (batch, num_uu, num_du) 的张量
        '''
        UUMat = self.UUInit(UUMat)  # (batch, num_uu, embed_dim*2)
        DUMat = self.DUInit(DUMat)  # (batch, num_du, embed_dim*2)
        INMat = self.INInit(INMat)  # (batch, num_in, embed_dim*2)
        TAMat = self.TAInit(TAMat)  # (batch, num_ta, embed_dim*2)
        CIMat = self.CIInit(CIMat.reshape(CIMat.shape[0], -1, 1)).reshape(CIMat.size(0), CIMat.size(1), CIMat.size(2))
        return UUMat, DUMat, INMat, TAMat, CIMat
    
if __name__ == "__main__":
    batch_size = 2
    embed_dim = 64
    num_trans = 8  # 发射天线数
    num_rece = 6   # 接收天线数
    
    # 设置每个矩阵的节点数量
    num_uu = 4    # UU节点数
    num_du = 3    # DU节点数
    num_in = 5    # IN节点数
    num_ta = 2    # TA节点数
    
    print("=== 测试 InitLayer 网络（复数张量）===")
    print(f"参数配置:")
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
    model = InitLayer(embed_dim, num_trans, num_rece)
    
    print(f"\n模型参数:")
    total_params = sum(p.numel() for p in model.parameters())
    print(f"总参数量: {total_params:,}")
    
    if(True):

        UU_init, DU_init, IN_init, TA_init, CI_init = model(UUMat, DUMat, INMat, TAMat, CIMat)
        print(f"\n输出张量维度:")
        print(f"UU_init: {UU_init.shape} (期望: [{batch_size}, {num_uu}, {embed_dim}])")
        print(f"DU_init: {DU_init.shape} (期望: [{batch_size}, {num_du}, {embed_dim}])")
        print(f"IN_init: {IN_init.shape} (期望: [{batch_size}, {num_in}, {embed_dim}])")
        print(f"TA_init: {TA_init.shape} (期望: [{batch_size}, {num_ta}, {embed_dim}])")
        print(f"CI_init: {CI_init.shape} (期望: [{batch_size}, {num_uu}, {num_du}])")
        
        # 验证输出形状
        assert UU_init.shape == (batch_size, num_uu, embed_dim)
        assert DU_init.shape == (batch_size, num_du, embed_dim)
        assert IN_init.shape == (batch_size, num_in, embed_dim)
        assert TA_init.shape == (batch_size, num_ta, embed_dim)
        assert CI_init.shape == (batch_size, num_uu, num_du)
        
        
       
        # 验证输出值范围
        print(f"\n输出实部值范围:")
        print(f"UU_init实部: [{UU_init.real.min().item():.4f}, {UU_init.real.max().item():.4f}]")
        print(f"DU_init实部: [{DU_init.real.min().item():.4f}, {DU_init.real.max().item():.4f}]")
        print(f"IN_init实部: [{IN_init.real.min().item():.4f}, {IN_init.real.max().item():.4f}]")
        print(f"TA_init实部: [{TA_init.real.min().item():.4f}, {TA_init.real.max().item():.4f}]")
        print(f"CI_init实部: [{CI_init.real.min().item():.4f}, {CI_init.real.max().item():.4f}]")
        
        print(f"\n输出虚部值范围:")
        print(f"UU_init虚部: [{UU_init.imag.min().item():.4f}, {UU_init.imag.max().item():.4f}]")
        print(f"DU_init虚部: [{DU_init.imag.min().item():.4f}, {DU_init.imag.max().item():.4f}]")
        print(f"IN_init虚部: [{IN_init.imag.min().item():.4f}, {IN_init.imag.max().item():.4f}]")
        print(f"TA_init虚部: [{TA_init.imag.min().item():.4f}, {TA_init.imag.max().item():.4f}]")
        print(f"CI_init虚部: [{CI_init.imag.min().item():.4f}, {CI_init.imag.max().item():.4f}]")
        
        # 验证输出确实是复数
        assert UU_init.dtype == torch.complex64
        assert DU_init.dtype == torch.complex64
        assert IN_init.dtype == torch.complex64
        assert TA_init.dtype == torch.complex64
        assert CI_init.dtype == torch.complex64
        
        print(f"\n✅ 所有输出均为复数张量!")