#假设已经得到嵌入后的节点UUMat, DUMat, INMat, TAMat
#不妨让以上矩阵的每一行对应一个节点
import numpy as np
import pandas as pd
import torch
import torch.nn as nn


#UU更新层
class RealUpdateLayer2UU(nn.Module):
    '''
    UU更新层:TA->UU, DU->UU, UU->UU, IN->UU
    '''
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        #定义参数和网络结构
        self.TA2UU=nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            batch_first=True
        )
        self.UU2UU=nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            batch_first=True
        )
        self.DU2UU=nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            batch_first=True
        )
        # 当将 CI 拼接到邻居节点特征后，使用 CI_fuse 将拼接后的维度映射回 embed_dim
        self.CI_fuse = nn.Linear(embed_dim + 2, embed_dim)
        self.IN2UU=nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            batch_first=True
        )
        self.DimUnit=nn.Linear(embed_dim*4, embed_dim)
        self.OutputUnit = nn.Sequential(
            nn.BatchNorm1d(embed_dim),
            nn.ReLU()  
        )
    def forward(self, UUMat, DUMat, INMat, TAMat, CIMat):
        du2uuList = []
        #对每个UU节点单独聚合特征
        for l in range(0, UUMat.size(1)):
            UUVec = UUMat[:, l, :]
            UUVec = UUVec.unsqueeze(1)
            # CIMat 已为实数表示，形状 (batch, num_uu, num_du, 2)
            # 取第 l 个 UU 与所有 DU 的 CI 向量 -> [batch, num_du, 2]
            CIVec = CIMat[:, l, :, :]
            # 拼接 CI 到 DUMat，然后用线性层映射回原始 embed_dim
            DUMatPlus = torch.cat([DUMat, CIVec], dim=2)
            DUMatPlus = self.CI_fuse(DUMatPlus)
            uu2duVec, _ = self.DU2UU(
                query=UUVec,
                key=DUMatPlus,
                value=DUMatPlus,
                need_weights=False
            )
            du2uuList.append(uu2duVec.squeeze(1))
        du2uu = torch.stack(du2uuList, dim=1)
        ta2uu, _ = self.TA2UU(
            query=UUMat,
            key=TAMat,
            value=TAMat,
            need_weights=False
        )
        uu2uu, _ = self.UU2UU(
            query=UUMat,
            key=UUMat,
            value=UUMat,
            need_weights=False
        )
        # IN -> UU: query should be UUMat
        in2uu, _ = self.IN2UU(
            query=UUMat,
            key=INMat,
            value=INMat,
            need_weights=False
        )
        uuConcat = torch.cat([du2uu, ta2uu, uu2uu, in2uu], dim=2)
        UUUpdated = self.DimUnit(uuConcat)
        UUUpdated = self.OutputUnit(UUUpdated.transpose(1,2)).transpose(1,2)
        return UUUpdated + UUMat
    


#DU更新层
class RealUpdateLayer2DU(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
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
        # 当将 CI 拼接到邻居节点特征后，使用 CI_fuse 将拼接后的维度映射回 embed_dim
        self.CI_fuse = nn.Linear(embed_dim + 2, embed_dim)
        self.DimUnit=nn.Linear(embed_dim*3, embed_dim)
        self.OutputUnit = nn.Sequential(
            nn.BatchNorm1d(embed_dim),
            nn.ReLU()  
        )
    def forward(self, UUMat, DUMat, TAMat, CIMat):
        uu2duList = []
        #对每个DU节点单独聚合特征
        for l in range(0, DUMat.size(1)):
            # 对第 l 个 DU 节点进行聚合
            DUVec = DUMat[:, l, :].unsqueeze(1)
            # CIMat 在本层为实数表示的转置，形状 (batch, num_du, num_uu, 2)
            CIVec = CIMat[:, l, :, :]  # [batch, num_uu, 2]
            # 拼接 CI 到 UUMat，然后映射回原始 embed_dim
            UUMatPlus = torch.cat([UUMat, CIVec], dim=2)
            UUMatPlus = self.CI_fuse(UUMatPlus)
            uu2duVec, _ = self.UU2DU(
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
        return DUUpdated+DUMat




#TA更新层
class RealUpdateLayer2TA(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        #定义参数和网络结构
        self.DU2TA=nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            batch_first=True
        )
        self.UU2TA=nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            batch_first=True
        )
        self.TA2TA=nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            batch_first=True
        )
        self.DimUnit=nn.Linear(embed_dim*3, embed_dim)
        self.OutputUnit = nn.Sequential(
            nn.BatchNorm1d(embed_dim),
            nn.ReLU()  
        )
    def forward(self, UUMat, DUMat, TAMat):
        du2ta, _ = self.DU2TA(
            query=TAMat,
            key=DUMat,
            value=DUMat,
            need_weights=False
        )
        uu2ta, _ = self.UU2TA(
            query=TAMat,
            key=UUMat,
            value=UUMat,
            need_weights=False
        )
        ta2ta, _ = self.TA2TA(
            query=TAMat,
            key=TAMat,
            value=TAMat,
            need_weights=False
        )
        taConcat = torch.cat([uu2ta, du2ta, ta2ta], dim=2)
        TAUpdated = self.DimUnit(taConcat)
        TAUpdated = self.OutputUnit(TAUpdated.transpose(1,2)).transpose(1,2)
        return TAUpdated+TAMat


#IN更新层
class RealUpdateLayer2IN(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        #定义参数和网络结构
        self.UU2IN=nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            batch_first=True
        )
        self.DimUnit=nn.Linear(embed_dim, embed_dim)
        self.OutputUnit = nn.Sequential(
            nn.BatchNorm1d(embed_dim),
            nn.ReLU()  
        )
    def forward(self, UUMat,INMat):
        uu2in, _ = self.UU2IN(
            query=INMat,
            key=UUMat,
            value=UUMat,
            need_weights=False
        )
        inConcat = uu2in
        INUpdated = self.DimUnit(inConcat)
        INUpdated = self.OutputUnit(INUpdated.transpose(1,2)).transpose(1,2)
        return INUpdated+INMat



#由于多头注意力机制不支持复数,我们需要将复数矩阵拆分为实部和虚部进行处理
class UpdateLayer(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        real_embed_dim = embed_dim * 2
        self.updateUU = RealUpdateLayer2UU(
            embed_dim=real_embed_dim, 
            num_heads=num_heads
        )
        self.updateDU = RealUpdateLayer2DU(
            embed_dim=real_embed_dim, 
            num_heads=num_heads
        )
        self.updateTA = RealUpdateLayer2TA(
            embed_dim=real_embed_dim, 
            num_heads=num_heads
        )
        self.updateIN = RealUpdateLayer2IN(
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
    def forward(self, UUMat_real, DUMat_real, INMat_real, TAMat_real, CIMat_real):
        """
        现在所有输入均为实数表示（若原始为复数，则实部和虚部已在最后一维拼接，
        因此节点特征的最后一维 = 2 * embed_dim；
        CIMat_real 的形状为 (batch, num_uu, num_du, 2) 表示 (real, imag)
        """
        # 直接将已转换的实数张量传入子层
        UUUpdated_real = self.updateUU(
            UUMat_real, DUMat_real, INMat_real, TAMat_real, CIMat_real
        )
        DUUpdated_real = self.updateDU(
            UUMat_real, DUMat_real, TAMat_real, CIMat_real.transpose(1,2)
        )
        TAUpdated_real = self.updateTA(
            UUMat_real, DUMat_real, TAMat_real
        )
        INUpdated_real = self.updateIN(
            UUMat_real, INMat_real
        )

        return UUUpdated_real, DUUpdated_real, TAUpdated_real, INUpdated_real

if __name__ == "__main__":
    def complex_to_real(t):
        return torch.cat([t.real, t.imag], dim=-1)
    def real_to_complex(x):
        f = x.shape[-1] // 2
        return torch.complex(x[..., :f], x[..., f:])
    torch.manual_seed(42)

    # 定义模型参数
    batch_size = 2
    num_uu = 3
    num_du = 4
    num_ta = 5
    num_in = 4
    embed_dim = 8
    num_heads = 4

    # 创建随机复数张量作为输入
    UUMat = torch.randn(batch_size, num_uu, embed_dim, dtype=torch.complex64)
    DUMat = torch.randn(batch_size, num_du, embed_dim, dtype=torch.complex64)
    TAMat = torch.randn(batch_size, num_ta, embed_dim, dtype=torch.complex64)
    CIMat = torch.randn(batch_size, num_uu, num_du, dtype=torch.complex64)
    INMat = torch.randn(batch_size, num_in, embed_dim, dtype=torch.complex64)

    

    UUMat_real = complex_to_real(UUMat)        # (batch, num_uu, 2*embed_dim)
    DUMat_real = complex_to_real(DUMat)        # (batch, num_du, 2*embed_dim)
    TAMat_real = complex_to_real(TAMat)        # (batch, num_ta, 2*embed_dim)
    INMat_real = complex_to_real(INMat)        # (batch, num_in, 2*embed_dim)
    # CIMat_real: (batch, num_uu, num_du, 2)
    CIMat_real = torch.stack([CIMat.real, CIMat.imag], dim=-1)

    # 打印输入张量信息
    print("输入张量形状:")
    print(f"UUMat: {UUMat.shape}")
    print(f"DUMat: {DUMat.shape}")
    print(f"TAMat: {TAMat.shape}")
    print(f"CIMat: {CIMat.shape}")
    print(f"INMat: {INMat.shape}")

    model = UpdateLayer(embed_dim, num_heads)
    # 打印模型结构
    print("\n模型结构:")
    print(model)

    UU_out_real, DU_out_real, TA_out_real, IN_out_real = model(
        UUMat_real, DUMat_real, INMat_real, TAMat_real, CIMat_real
    )

    # 主程序把实数输出还原为复数
    UU_out = real_to_complex(UU_out_real)
    DU_out = real_to_complex(DU_out_real)
    TA_out = real_to_complex(TA_out_real)
    IN_out = real_to_complex(IN_out_real)

    # 检查输出
    print("\n输出张量信息:")
    print(f"UU_out 形状: {UU_out.shape}, is_complex: {UU_out.is_complex()}")
    print(f"DU_out 形状: {DU_out.shape}, is_complex: {DU_out.is_complex()}")
    print(f"TA_out 形状: {TA_out.shape}, is_complex: {TA_out.is_complex()}")
    print(f"IN_out 形状: {IN_out.shape}, is_complex: {IN_out.is_complex()}")