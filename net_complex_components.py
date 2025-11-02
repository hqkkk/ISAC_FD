import torch
import torch.nn as nn
import net_complex_basis as ncb
import torch.nn.functional as F
class MultiheadAttention(nn.Module):
    '''
    多头注意力机制
    输入：
        batch_size, Qnum, dim_qk = query.size()
        batch_size, Knum, dim_qk = key.size()
        batch_size, Vnum, dim_v = value.size()
    输出：
        batch_size, Qnum, num_heads*self.dim_v
    '''
    def __init__(self, dim_qk, dim_v, num_heads, batch_first=True):
        super().__init__()
        self.WQ = ncb.linear(dim_qk, num_heads*dim_qk)
        self.WK = ncb.linear(dim_qk, num_heads*dim_qk)
        self.WV = ncb.linear(dim_v, num_heads*dim_v)
        self.dk = dim_qk
        self.num_heads = num_heads
        self.dim_v = dim_v
    def forward(self, query, key, value, need_weights=False):  
        batch_size, Qnum, Qdim = query.size()
        batch_size, Knum, Kdim = key.size()
        batch_size, Vnum, dim = value.size()
        Q = self.WQ(query).view(batch_size, Qnum, self.num_heads, self.dk).permute(0, 2, 1, 3)
        K = self.WK(key).view(batch_size, Knum, self.num_heads, self.dk).permute(0, 2, 3, 1)  # 这里完成转置
        V = self.WV(value).view(batch_size, Vnum, self.num_heads, self.dim_v).permute(0, 2, 1, 3)

        attention_score = torch.abs(torch.matmul(Q, K)) / torch.sqrt(torch.tensor(float(self.dk)))
        attention_weight = F.softmax(attention_score, dim=-1) + 0*1j

        output = torch.matmul(attention_weight, V).permute(0, 2, 1, 3).reshape(batch_size, Qnum, self.num_heads*self.dim_v)

        return output

class DNN_3layer(nn.Module):
    '''
    三层全连接神经网络
    输入：
        batch_size, num, feature_in
    输出:
        batch_size, num, feature_out
    '''
    def __init__(self, feature_in, feature_hidden, feature_out):
        super().__init__()
        self.layer1 = ncb.linear(feature_in, feature_hidden)
        self.layer2 = ncb.linear(feature_hidden, feature_hidden)
        self.layer3 = ncb.linear(feature_hidden, feature_out)
        self.relu = ncb.relu()
        self.bn1 = ncb.NaiveComplexBatchNorm1d(feature_hidden)
        self.bn2 = ncb.NaiveComplexBatchNorm1d(feature_out)
        self.sigmoid = ncb.sigmoid()
    def forward(self, x):
        '''
        输入：
            x: 形状为 (batch_size, num, feature_in) 的复数张量
        输出:
            out: 形状为 (batch_size, num, feature_out) 的复数张量
        '''
        x = self.layer1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer2(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer3(x)
        x = self.bn2(x)
        x = self.sigmoid(x)

        return x
    
class complexpost(nn.Module):
    '''
    复数后处理模块
    输入:
        batch_size, num, feature
    输出:
        batch_size, num, feature
    '''
    def __init__(self, in_size, out_size, activation="sigmoid"):
        super().__init__()
        self.dimUnit=ncb.linear(in_size, out_size)
        if(activation=="sigmoid"):
            self.activation=ncb.sigmoid()
        elif(activation=="relu"):
            self.activation=ncb.relu()
        self.bn=ncb.NaiveComplexBatchNorm1d(out_size)
    def forward(self, x):
        '''
        输入：
            x: 形状为 (batch_size, num, feature) 的复数张量
        输出:
            out: 形状为 (batch_size, num, feature) 的复数张量
        '''
        x = self.dimUnit(x)
        x = self.bn(x)
        x = self.activation(x)
        return x