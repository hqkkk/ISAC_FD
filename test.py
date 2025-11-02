import torch

def batch_quadratic_form(G, Q):
    """
    批量计算 g_l^H Q g_l
    
    参数:
        G: 形状为 (N, L) 的矩阵，每列是一个向量 g_l
        Q: 形状为 (N, N) 的矩阵
    
    返回:
        result: 形状为 (L,) 的张量，包含每个 g_l^H Q g_l 的结果
    """
    # 计算 QG = Q @ G，形状 (N, L)
    QG = Q @ G
    
    # 计算 G^H @ (Q @ G) 的对角线元素，即每个 g_l^H (Q g_l)
    # 使用逐元素乘法和求和
    result = torch.sum(G.conj() * QG, dim=0)
    
    return result

# 示例使用
N, L = 100, 50  # 向量维度N，向量数量L
G = torch.randn(N, L, dtype=torch.complex64)
Q = torch.randn(N, N, dtype=torch.complex64)

result = batch_quadratic_form(G, Q)
print(f"结果形状: {result.shape}")