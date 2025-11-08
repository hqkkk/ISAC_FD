import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import netAll
import torch_math_complex as tmc
def geta(angleArr, num):
    '''
    生成导向矢量矩阵
    angleArr: (batch, m) 角度数组（度）
    num: 天线数量
    返回: (batch, m, num) 复数导向矢量
    '''
    batch, m = angleArr.shape
    # 天线索引
    n = torch.arange(num, dtype=torch.float32)  # 形状: (num,)
    # 角度转换为弧度并计算sin
    sin_theta = torch.sin(angleArr * torch.pi / 180)  # 形状: (batch, m)
    # 使用einsum进行计算
    # 计算: π * n_i * sin(θ_j) 对于每个批次
    phase = torch.einsum('i,bj->bij', n, sin_theta) * torch.pi  # 形状: (batch, m, num)
    # 复数导向矢量
    a = torch.exp(1j * phase)  # 形状: (batch, m, num)
    return a

class LossFunction(nn.Module):
    def __init__(self, lambda1=100):
        super().__init__()
        self.lambda1 = lambda1
    def forward(self, UUPowerMat, DUComMat, SensingMat, 
                 UUMatInit, DUMatInit, TAMatInit, INMatInit, 
                 CIMatInit, noise2DU, noise2BS, alpha_SI,num_trans,num_rece,
                 angle_delta_degrees = 10, angle_precision_degrees = 1):
        batch_size = UUPowerMat.shape[0]
        M = TAMatInit.shape[1] 
        K = UUMatInit.shape[1]
        I = INMatInit.shape[1]
        L = DUMatInit.shape[1]
        N_t = num_trans
        N_r = num_rece

        alpha_SI_linear = 10**(alpha_SI / 10)
        noise2BS = 10**(noise2BS / 10)
        noise2DU = 10**(noise2DU / 10)
        # 生成随机相位矩阵，均匀分布在[0, 2π)
        random_phase = 2 * torch.pi * torch.rand(N_r, N_t, device=device)
    
        # 构建H_SI矩阵：幅度为sqrt(alpha_SI_linear)，相位为随机相位
        # 注意：由于是功率，幅度需要开方
        myH_SI = torch.sqrt(alpha_SI_linear) * torch.exp(-1j * random_phase) 
        #(batch, M)
        TAPowerGainArr = TAMatInit[:,:,1].real
        TAangleArr = TAMatInit[:,:,0].real
        #angleArr: (batch, M+INnum)
        angleArr = torch.cat([TAMatInit[:,:,0].real,INMatInit[:,:,0].real],dim=1)
        #powerGainArr: (batch, M+INnum)
        powerGainArr = torch.cat([TAMatInit[:,:,1].real,INMatInit[:,:,1].real],dim=1)
        #x(batch,N_t,N_t)
        #Q: (batch, N_t, N_t)
        #a_t: (batch, M+I, N_t)
        #a_r: (batch, M+I, N_r)
        
        x = DUComMat.sum(dim=1) + SensingMat.sum(dim=1)
        Q = tmc.getHermitian(x)
        a_t = geta(angleArr,num_trans)
        a_r = geta(angleArr,num_rece)
                   
        B = torch.zeros((batch_size, N_r, N_t), dtype=torch.complex64)
        for l in range(M+I):
        # a_r[：,:,l] shape: (batch, N_r, 1), a_t[：,:,l].conj().T shape: (batch, 1, N_t)
        # 外积结果 shape: (batch, N_r, N_t)
        #powerGainArr: (batch, M+INnum)
            #print('a_r shape', a_r.shape, 'a_t shape', a_t.shape)
            #print('ar_l', a_r[:, l, :].shape, 'at_l', a_t[:, l, :].shape)
            
            outer_product = powerGainArr[:, l].view(-1,1,1) * (a_r[:, l, :].unsqueeze(2) @ a_t[:, l, :].conj().unsqueeze(1))
            B += outer_product
            #print('outer', outer_product.shape)
        B += myH_SI
        
        #NoiseBSMat:(batch, N_r, N_r)
        NoiseBSMat = torch.eye(num_rece, dtype=torch.complex64) * noise2BS
        stacked_list = torch.stack([NoiseBSMat.clone() for _ in range(batch_size)], dim=0)
        #NoiseBSMat = NoiseBSMat.unsqueeze(0).repeat(batch_size, 1, 1)
        
        #hk_H_list: list of (batch,N_r, N_r)
        #sum_hk: (batch, N_r, N_r)
        sum_hk = torch.zeros((batch_size, num_rece, num_rece), dtype=torch.complex64)
        hk_H_list = []
        for l in range(K):
        # a_r[:,l] shape: (N_r, 1), a_t[:,l].conj().T shape: (1, N_t)
        # 外积结果 shape: (N_r, N_t)
            outer_product = UUPowerMat[:,l].view(-1,1,1)*(UUMatInit[:, l, :].unsqueeze(2) @ UUMatInit[:, l, :].conj().unsqueeze(1))
            sum_hk += outer_product
            hk_H_list.append(outer_product)
        uu_SINR_denominator_inside = sum_hk + NoiseBSMat + B @ Q @ B.conj().permute(0,2,1)

        #wk_list: list of (batch, N_r, N_r)
        wk_list = []
        for l in range(K):
            #item_sum: (batch, N_r, N_r)
            item_sum = torch.linalg.inv(uu_SINR_denominator_inside - hk_H_list[l])
            item_sum = item_sum @ UUMatInit[:, l, :].unsqueeze(2)
            #item_sum: (batch, N_r, 1)
            wk_list.append(item_sum)
        #每批样本中第k个上行用户SINR
        uu_SINR = []
        for l in range(K):  
            uu_SINR_numerator = wk_list[l].conj().permute(0,2,1) @ hk_H_list[l] @ wk_list[l]
            uu_SINR_denominator = wk_list[l].conj().permute(0,2,1) @ uu_SINR_denominator_inside @ wk_list[l]
            uu_SINR.append(uu_SINR_numerator / uu_SINR_denominator)
        uu_SINR_tensor = torch.stack(uu_SINR, dim=1)  # 形状: (batch_size, K)
        # 计算和速率（所有用户的速率之和）
        sum_rate_uu = torch.log2(1.0 + uu_SINR_tensor.real).sum(dim=1)
        
        #求下行
        gHQg_Mat = torch.einsum('bij,bjk,bik->bi', DUMatInit.conj(), Q, DUMatInit)
        # 将结果reshape为 (batch, num_du, 1)
        gHQg_Mat = gHQg_Mat.unsqueeze(2)
        #torch.abs(CIMatInit) ** 2: (batch, K, L)
        #UUPowerMat: (batch, K, 1)
        #noiseUU2DU:(batch, L,1)
        noiseUU2DU = UUPowerMat.unsqueeze(2).permute(0,2,1) @ (torch.abs(CIMatInit) ** 2) 
        du_SINR=[]
        for l in range(L):
            du_SINR_numerator = gHQg_Mat[:,l,:].unsqueeze(2)
            #noiseUU2DU.[]:(batch, 1,1) noise2DU:(batch, 1,1)
            #du_SINR_denominator:(batch, 1,1)
            du_SINR_denominator = noiseUU2DU[:,:,l].unsqueeze(2) + noise2DU[:,l,:].unsqueeze(2) + gHQg_Mat.sum(dim=1,keepdim=True) - gHQg_Mat[:,l,:].unsqueeze(2)
            du_SINR.append(du_SINR_numerator / du_SINR_denominator) 
        du_SINR_tensor = torch.stack(du_SINR, dim=1)  # 形状: (batch_size, K)
        # 计算和速率（所有用户的速率之和）
        sum_rate_du = torch.log2(1.0 + du_SINR_tensor.real).sum(dim=1)

        angle_num = 180/angle_precision_degrees
        angle_grid = torch.linspace(0, 180, int(angle_num+1))
        angle_grid_len = angle_num+1
        #设定是否在区间内
        angle_expanded = TAangleArr.unsqueeze(2)  # 形状: (batch_size, M, 1)
        lower_bounds = angle_expanded - angle_delta_degrees  # 形状: (batch_size, M, 1)
        upper_bounds = angle_expanded + angle_delta_degrees  # 形状: (batch_size, M, 1)

        angle_grid_expanded = angle_grid.unsqueeze(0).unsqueeze(0)  # 形状: (1, 1, angle_grid_len)

        # 批量比较：判断每个网格点是否在每个角度的区间内
        # indices形状: (batch_size, M, angle_grid_len)
        indices = (angle_grid_expanded >= lower_bounds) & (angle_grid_expanded <= upper_bounds)

        # 沿M维度进行逻辑或，合并同一批次中所有角度的区间
        b_theta = indices.any(dim=1)  # 形状: (batch_size, angle_grid_len)
        #a_t_grid: (batch, angle_grid_len, N_t)
        a_t_grid = geta(angle_grid.unsqueeze(0).repeat(batch_size,1),num_trans)
        P = torch.einsum('bim,bij,bjm->bm', a_t_grid.conj(), Q, a_t_grid)  # 形状: (batch, angle_grid_len)

        # 计算 b^H b (形状: (batch, 1))
        bH_b = torch.sum(b_theta.conj() * b_theta, dim=1, keepdim=True).real  # 取实部
    
        # 计算 b^H P (形状: (batch, 1))
        bH_P = torch.sum(b_theta.conj() * P, dim=1, keepdim=True)
    
        # 计算最优β: β = (b^H b)^{-1} b^H P
        # 添加小常数防止除零
        beta_opt = bH_P / (bH_b + 1e-10)
    
        # 计算最小二乘损失: L = (1/N) * Σ|βb - P|²
        diff = beta_opt * b_theta - P
        L_opt = torch.sum(torch.abs(diff)**2, dim=1, keepdim=True) / angle_grid_len
        loss = - (sum_rate_uu + sum_rate_du).mean() +self.lambda1*L_opt.mean()
        return loss
