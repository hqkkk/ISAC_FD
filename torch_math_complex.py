import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np

def getHermitian(x):
    """
    计算列向量 x 与 xᴴ 的外积
    输入: x 形状[batch, n] 的复数张量
    输出: 形状为 [batch, n, n] 的矩阵
    """
    result = torch.einsum('bi,bj->bij', x, x.conj())
    
    return result