import torch
import torch.nn as nn
import net_complex_basis as ncb
import torch.nn.functional as F
class MultiheadAttention(nn.module):
    '''
    多头注意力机制
    输入：
        batch_size, Qnum, dim_qk = forQ.size()
        batch_size, Knum, dim_qk = forK.size()
        batch_size, Vnum, dim_v = forV.size()
    输出：
        batch_size, Qnum, num_heads*self.dim_v
    '''
    def __init__(self, dim_qk, dim_v, num_heads, feature ):
        super().__init__()
        self.WQ = ncb.linear(feature, num_heads*dim_qk)
        self.WK = ncb.linear(feature, num_heads*dim_qk)
        self.WV = ncb.linear(feature, num_heads*dim_v)
        self.dk = dim_qk
        self.num_heads = num_heads
        self.dim_v = dim_v
    def forward(self, forQ, forK, forV):  
        batch_size, Qnum, dim = forQ.size()
        batch_size, Knum, dim = forK.size()
        batch_size, Vnum, dim = forV.size()
        if(Knum != Vnum):
            raise ValueError("Knum must be equal to Vnum")
        Q = self.WQ(forQ).view(batch_size, Qnum, self.num_heads, self.size).permute(0, 2, 1, 3)
        K = self.WK(forK).view(batch_size, Knum, self.num_heads, self.size).permute(0, 2, 3, 1)  # 这里完成转置
        V = self.WV(forV).view(batch_size, Vnum, self.num_heads, self.dim_v).permute(0, 2, 1, 3)

        attention_score = torch.abs(torch.matmul(Q, K)) / torch.sqrt(torch.tensor(float(self.dk)))
        attention_weight = F.softmax(attention_score, dim=-1) + 0*1j

        output = torch.matmul(attention_weight, V).permute(0, 2, 1, 3).reshape(batch_size, Qnum, self.num_heads*self.dim_v)

        return output
