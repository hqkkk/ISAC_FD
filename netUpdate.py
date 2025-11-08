#假设已经得到嵌入后的节点UUMat, DUMat, INMat, TAMat
#不妨让以上矩阵的每一行对应一个节点
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import net_complex_basis as ncb
import net_complex_components as ncc

#UU更新层
class UpdateLayer2UU(nn.Module):
    '''
    UU更新层:TA->UU, DU->UU, UU->UU, IN->UU
    '''
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        #定义参数和网络结构
        self.TA2UU=ncc.MultiheadAttention(
            dim_qk=embed_dim,
            dim_v=embed_dim,
            num_heads=num_heads,
            batch_first=True
        )
        self.UU2UU=ncc.MultiheadAttention(
             dim_qk=embed_dim,
            dim_v=embed_dim,
            num_heads=num_heads,
            batch_first=True
        )
        self.DU2UU=ncc.MultiheadAttention(
             dim_qk=embed_dim,
            dim_v=embed_dim,
            num_heads=num_heads,
            batch_first=True
        )
        self.IN2UU=ncc.MultiheadAttention(
             dim_qk=embed_dim,
            dim_v=embed_dim,
            num_heads=num_heads,
            batch_first=True
        )
        self.CI_fuse = ncb.linear(embed_dim + 1, embed_dim)
        self.resnet = ncb.resnet()
        self.Wo = ncb.linear(embed_dim*4, embed_dim)
        self.post = ncc.complexpost(embed_dim, embed_dim)
    def forward(self, UUMat, DUMat, INMat, TAMat, CIMat):
        du2uuList = []
        #对每个UU节点单独聚合特征
        for l in range(0, UUMat.size(1)):
            # 对第 l 个 UU 节点进行聚合
            UUVec = UUMat[:, l, :].unsqueeze(1)# [batch,1,embed_dim]
            # CIMat 在本层为复数表示，形状 (batch, num_uu, num_du)
            CIVec = CIMat[:, l, :].unsqueeze(1)# [batch,1,num_du]
            # 拼接 CI 到 DUMat，然后映射回原始 embed_dim
            DUMatPlus = torch.cat([DUMat, CIVec.permute(0,2,1)], dim=2)
            DUMatPlus = self.CI_fuse(DUMatPlus)
            uu2duVec = self.DU2UU(
                query=UUVec,
                key=DUMatPlus,
                value=DUMatPlus,
            )
            du2uuList.append(uu2duVec.squeeze(1))
        du2uu = torch.stack(du2uuList, dim=1)
        ta2uu = self.TA2UU(
            query=UUMat,
            key=TAMat,
            value=TAMat,
            need_weights=False
        )
        uu2uu = self.UU2UU(
            query=UUMat,
            key=UUMat,
            value=UUMat,
            need_weights=False
        )
        # IN -> UU: query should be UUMat
        in2uu = self.IN2UU(
            query=UUMat,
            key=INMat,
            value=INMat,
            need_weights=False
        )
        uuConcat = torch.cat([du2uu, ta2uu, uu2uu, in2uu], dim=2)
        uuConcat = self.Wo(uuConcat)
        uumat = self.resnet(UUMat, uuConcat)
        UUMat = self.post(uumat)
        return UUMat
    


#DU更新层
class UpdateLayer2DU(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        #定义参数和网络结构
        self.TA2DU=ncc.MultiheadAttention(
            dim_qk=embed_dim,
            dim_v=embed_dim,
            num_heads=num_heads,
            batch_first=True
        )
        self.UU2DU=ncc.MultiheadAttention(
            dim_qk=embed_dim,
            dim_v=embed_dim,
            num_heads=num_heads,
            batch_first=True
        )
        self.DU2DU=ncc.MultiheadAttention(
            dim_qk=embed_dim,
            dim_v=embed_dim,
            num_heads=num_heads,
            batch_first=True
        )
        # 当将 CI 拼接到邻居节点特征后，使用 CI_fuse 将拼接后的维度映射回 embed_dim
        self.CI_fuse = ncb.linear(embed_dim + 1, embed_dim)
        self.resnet = ncb.resnet(embed_dim,embed_dim*num_heads*3)
        self.Wo = ncb.linear(embed_dim*3, embed_dim)
        self.post = ncc.complexpost(embed_dim, embed_dim)
    def forward(self, UUMat, DUMat, TAMat, CIMat):
        uu2duList = []
        #对每个DU节点单独聚合特征
        for l in range(0, DUMat.size(1)):
            # 对第 l 个 DU 节点进行聚合
            DUVec = DUMat[:, l, :].unsqueeze(1)#batch,1,embed_dim
            # CIMat 在本层为实数表示的转置，形状 (batch, uu, du)
            CIVec = CIMat[:, :, l].unsqueeze(2)  # [batch, num_uu,1]
            # 拼接 CI 到 UUMat，然后映射回原始 embed_dim
            UUMatPlus = torch.cat([UUMat, CIVec], dim=2)#[baatch,num_uu,embed_dim+1]
            UUMatPlus = self.CI_fuse(UUMatPlus)# [batch,num_uu,embed_dim]
            uu2duVec = self.UU2DU(
                query=DUVec,
                key=UUMatPlus,
                value=UUMatPlus,
                need_weights=False
            )
            uu2duList.append(uu2duVec.squeeze(1))
        uu2du = torch.stack(uu2duList, dim=1)
        ta2du = self.TA2DU(
            query=DUMat,
            key=TAMat,
            value=TAMat,
            need_weights=False
        )
        du2du = self.DU2DU(
            query=DUMat,
            key=DUMat,
            value=DUMat,
            need_weights=False
        )
        duConcat = torch.cat([uu2du, ta2du, du2du], dim=2)
        duConcat = self.Wo(duConcat)
        dumat = self.resnet(DUMat, duConcat)
        DUMat = self.post(dumat)
        return DUMat




#TA更新层
class UpdateLayer2TA(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        #定义参数和网络结构
        self.DU2TA=ncc.MultiheadAttention(
            dim_qk=embed_dim,
            dim_v=embed_dim,
            num_heads=num_heads,
            batch_first=True
        )
        self.UU2TA=ncc.MultiheadAttention(
            dim_qk=embed_dim,
            dim_v=embed_dim,
            num_heads=num_heads,
            batch_first=True
        )
        self.TA2TA=ncc.MultiheadAttention(
            dim_qk=embed_dim,
            dim_v=embed_dim,
            num_heads=num_heads,
            batch_first=True
        )
        self.post = ncc.complexpost(embed_dim, embed_dim)
        self.Wo = ncb.linear(embed_dim*3, embed_dim)
        self.resnet = ncb.resnet(embed_dim,embed_dim*num_heads*3)
    def forward(self, UUMat, DUMat, TAMat):
        du2ta = self.DU2TA(
            query=TAMat,
            key=DUMat,
            value=DUMat,
            need_weights=False
        )
        uu2ta = self.UU2TA(
            query=TAMat,
            key=UUMat,
            value=UUMat,
            need_weights=False
        )
        ta2ta = self.TA2TA(
            query=TAMat,
            key=TAMat,
            value=TAMat,
            need_weights=False
        )
        taConcat = torch.cat([uu2ta, du2ta, ta2ta], dim=2)
        taConcat = self.Wo(taConcat)
        tamat = self.resnet(TAMat, taConcat)
        TAMat = self.post(tamat)
        return TAMat


#IN更新层
class UpdateLayer2IN(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        #定义参数和网络结构
        self.UU2IN=ncc.MultiheadAttention(
            dim_qk=embed_dim,
            dim_v=embed_dim,
            num_heads=num_heads,
            batch_first=True
        )
        self.post = ncc.complexpost(embed_dim, embed_dim)
        self.Wo = ncb.linear(embed_dim, embed_dim)
        self.resnet = ncb.resnet(embed_dim,embed_dim*num_heads)
    def forward(self, UUMat,INMat):
        uu2in = self.UU2IN(
            query=INMat,
            key=UUMat,
            value=UUMat,
            need_weights=False
        )
        inConcat = uu2in
        inConcat = self.Wo(inConcat)
        inmat = self.resnet(INMat, inConcat)
        INMat = self.post(inmat)
        return INMat


class UpdateLayer(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.updateUU=UpdateLayer2UU(embed_dim, num_heads)
        self.updateDU=UpdateLayer2DU(embed_dim, num_heads)
        self.updateTA=UpdateLayer2TA(embed_dim, num_heads)
        self.updateIN=UpdateLayer2IN(embed_dim, num_heads)
    def forward(self, UUMat, DUMat, INMat, TAMat, CIMat):
        # 直接将张量传入子层
        UUUpdated = self.updateUU(
            UUMat, DUMat, INMat, TAMat, CIMat
        )
        DUUpdated = self.updateDU(
            UUMat, DUMat, TAMat, CIMat
        )
        TAUpdated = self.updateTA(
            UUMat, DUMat, TAMat
        )
        INUpdated = self.updateIN(
            UUMat, INMat
        )

        return UUUpdated, DUUpdated, TAUpdated, INUpdated

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
    
    # 创建随机初始化的输入张量
    UUMat = torch.randn(batch_size, num_uu, embed_dim,dtype=torch.complex64)  # (4, 10, 64)
    DUMat = torch.randn(batch_size, num_du, embed_dim,dtype=torch.complex64)  # (4, 6, 64)
    INMat = torch.randn(batch_size, num_in, embed_dim,dtype=torch.complex64)  # (4, 8, 64)
    TAMat = torch.randn(batch_size, num_ta, embed_dim,dtype=torch.complex64)  # (4, 5, 64)
    CIMat = torch.randn(batch_size, num_uu, num_du,dtype=torch.complex64)  # (4,10,6)
    
    # 创建模型
    model = UpdateLayer(embed_dim, num_heads)
    
    # 前向传播
    UUUpdated, DUUpdated, TAUpdated, INUpdated = model(UUMat, DUMat, INMat, TAMat, CIMat)
    
    # 检查输出维度
    print("输入维度:")
    print(f"UUMat: {UUMat.shape}")
    print(f"DUMat: {DUMat.shape}") 
    print(f"INMat: {INMat.shape}")
    print(f"TAMat: {TAMat.shape}")
    print(f"CIMat: {CIMat.shape}")
    
    print("\n输出维度:")
    print(f"UUUpdated: {UUUpdated.shape} (应与UUMat相同: [{batch_size}, {num_uu}, {embed_dim}])")
    print(f"DUUpdated: {DUUpdated.shape} (应与DUMat相同: [{batch_size}, {num_du}, {embed_dim}])")
    print(f"TAUpdated: {TAUpdated.shape} (应与TAMat相同: [{batch_size}, {num_ta}, {embed_dim}])")
    print(f"INUpdated: {INUpdated.shape} (应与INMat相同: [{batch_size}, {num_in}, {embed_dim}])")
    
    # 验证输出形状是否正确
    assert UUUpdated.shape == (batch_size, num_uu, embed_dim)
    assert DUUpdated.shape == (batch_size, num_du, embed_dim)
    assert TAUpdated.shape == (batch_size, num_ta, embed_dim)
    assert INUpdated.shape == (batch_size, num_in, embed_dim)
    
    print("\n✅ 所有测试通过！模型运行正常。")
