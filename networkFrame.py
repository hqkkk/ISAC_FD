#假设已经得到嵌入后的节点UUMat, DUMat, INMat, TAMat
#不妨让以上矩阵的每一列对应一个节点
import numpy as np
import pandas as pd
import torch
import torch.nn as nn



class RealUpdateLayer2DU(nn.Module):
    def __init__(self, UUMat, DUMat, INMat,TAMat,embed_dim, num_heads):
        super().__init__()
        self.UUMat = UUMat
        self.DUMat = DUMat
        self.INMat = INMat
        self.TAMat = TAMat
        #定义参数和网络结构
        self.TA2DU=nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            batch_first=True
        )
        self.UU2DU=nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            batch_first=True
        )
        self.DU2DU=nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            batch_first=True
        )
        self.DimUnit=nn.Linear(embed_dim*3, embed_dim)
        self.OutputUnit = nn.Sequential(
            nn.BatchNorm1d(embed_dim),
            nn.ReLU()  
        )
    def forward(self, UUMat, DUMat, TAMat, CIMat):
        uu2duList = []
        #对每个DU节点单独聚合特征
        for l in range(0,DUMat.size(1)):
            #complex:CI->batchsize*K*L h_{k,l}:k-th UU->l-th DU
            DUVec = DUMat[:,l,:];
            DUVec = DUVec.unsqueeze(1);
            real_part = CIMat[:, :, l]            # [batch, num_uu]
            imag_part = CIMat[:, :, l + DUMat.size(1)]   # [batch, num_uu]
            # 将实部和虚部组合成2维向量
            CIVec = torch.stack([real_part, imag_part], dim=-1)  # [batch, num_uu, 2]
            #UUMat->batchsize*K*feature,UUMatPlus->batchsize*K*(feature+2)
            UUMatPlus = torch.cat([UUMat, CIVec], dim=2)
            uu2duVec,_ = self.UU2DU(
                query=DUVec,
                key=UUMatPlus,
                value=UUMatPlus,
                need_weights=False
            )
            uu2duList.append(uu2duVec.squeeze(1))
        uu2du = torch.stack(uu2duList, dim=1)
        ta2du, _ = self.TA2DU(
            query=DUMat,
            key=TAMat,
            value=TAMat,
            need_weights=False
        )
        du2du, _ = self.DU2DU(
            query=DUMat,
            key=DUMat,
            value=DUMat,
            need_weights=False
        )
        duConcat = torch.cat([uu2du, ta2du, du2du], dim=2)
        DUUpdated = self.DimUnit(duConcat)
        DUUpdated = self.OutputUnit(DUUpdated.transpose(1,2)).transpose(1,2)
        return DUUpdated

#由于多头注意力机制不支持复数,我们需要将复数矩阵拆分为实部和虚部进行处理
class UpdateLayer2DU(nn.Module):
    def __init__(self, UUMat, DUMat, INMat, TAMat, embed_dim, num_heads):
        super().__init__()
        real_embed_dim = embed_dim * 2
        self.update_layer = RealUpdateLayer2DU(
            UUMat, DUMat, INMat, TAMat, 
            embed_dim=real_embed_dim, 
            num_heads=num_heads
        )
    
    def Complex2Real(self, tensor):
        return torch.cat([tensor.real, tensor.imag], dim=-1)
    
    def Real2Complex(self, real_tensor):
        features = real_tensor.shape[-1] // 2
        real_part = real_tensor[..., :features]
        imag_part = real_tensor[..., features:]
        return torch.complex(real_part, imag_part)
    
    def forward(self, UUMat, DUMat, TAMat, CIMat):
        # 将输入转换为实数表示
        UUMat_real = self.Complex2Real(UUMat)
        DUMat_real = self.Complex2Real(DUMat)
        TAMat_real = self.Complex2Real(TAMat)
        CIMat_real = self.Complex2Real(CIMat)
        
        # 通过更新层处理实数张量
        DUUpdated_real = self.update_layer(
            UUMat_real, DUMat_real, TAMat_real, CIMat_real
        )

        return self.Real2Complex(DUUpdated_real)

if __name__ == "__main__":
    torch.manual_seed(42)
    
    # 定义模型参数
    batch_size = 2
    num_uu = 3  
    num_du = 4  
    num_ta = 5  
    embed_dim = 8  
    num_heads = 4
    
    # 创建随机复数张量作为输入
    UUMat = torch.randn(batch_size, num_uu, embed_dim-1, dtype=torch.complex64)
    DUMat = torch.randn(batch_size, num_du, embed_dim, dtype=torch.complex64)
    TAMat = torch.randn(batch_size, num_ta, embed_dim, dtype=torch.complex64)
    CIMat = torch.randn(batch_size, num_uu, num_du, dtype=torch.complex64)
    
    # 打印输入张量信息
    print("输入张量形状:")
    print(f"UUMat: {UUMat.shape}")
    print(f"DUMat: {DUMat.shape}")
    print(f"TAMat: {TAMat.shape}")
    print(f"CIMat: {CIMat.shape}")
    
    # 初始化模型
    model = UpdateLayer2DU(
        UUMat=None,  # 这些参数在forward中传入，所以这里可以设为None
        DUMat=None,
        INMat=None,
        TAMat=None,
        embed_dim=embed_dim,
        num_heads=num_heads
    )
    
    # 打印模型结构
    print("\n模型结构:")
    print(model)
    
    # 前向传播
    output = model(UUMat, DUMat, TAMat, CIMat)
    
    # 检查输出
    print("\n输出张量信息:")
    print(f"形状: {output.shape}")
    print(f"数据类型: {output.dtype}")
    print(f"是否为复数: {output.is_complex()}")
    
    # 验证输出维度
    assert output.shape == (batch_size, num_du, embed_dim), "输出形状不正确!"
    assert output.is_complex(), "输出不是复数张量!"
    
    print("\n测试通过!模型功能正常。")