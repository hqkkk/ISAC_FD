#用于定义网络组件
import torch
import torch.nn as nn
import torch.nn.functional as F


class linear(nn.Module):
    '''
    复数线性层
    输入:(batch, sequence, feature)
    Y = WX + b
    '''
    def __init__(self, in_size, out_size, bias=False):
        super().__init__()
        self.fc = nn.Linear(in_size, out_size, dtype=torch.complex64)
        self.bias = bias
        if(bias == True):
            self.fc.bias = nn.Parameter(torch.randn(out_size, dtype=torch.complex64))
    def forward(self, x):
        x=x+0j
        if(self.bias == True):
            x = x + self.fc.bias
            return x
        x = self.fc(x)
        return x

class relu(nn.Module):
    '''
    输入:(batch, sequence, feature)
    '''
    def __init__(self):
        super().__init__()
    def forward(self, x):
        x = torch.relu(x.real) + 1j*torch.relu(x.imag)
        return x

class tanh(nn.Module):
    '''
    输入:(batch, sequence, feature)
    '''
    def __init__(self):
        super().__init__()
    def forward(self, x):
        x = torch.tanh(x)
        return x

class sigmoid(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        real = x.real
        imag = x.imag
        real = F.sigmoid(real)
        imag = F.sigmoid(imag)
        return real + imag * 1j

class NaiveComplexBatchNorm1d(nn.Module):
    """
    朴素复数批归一化：分别对实部和虚部进行归一化
    输入: (batch, sequence, features)
    输出: (batch, sequence, features)
    """
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        # 实部和虚部分别使用独立的BatchNorm
        self.bn_real = nn.BatchNorm1d(num_features, eps=eps, momentum=momentum)
        self.bn_imag = nn.BatchNorm1d(num_features, eps=eps, momentum=momentum)
    
    def forward(self, x):
        batch_size, seq_len, num_features = x.shape
        # 分离实部和虚部
        real_part = x.real  # (batch, sequence, features)
        imag_part = x.imag  # (batch, sequence, features)
        
        # 调整形状(batch, features, sequence)
        real_part = real_part.transpose(1, 2)  # (batch, features, sequence)
        imag_part = imag_part.transpose(1, 2)  # (batch, features, sequence)
        
        # 分别归一化
        norm_real = self.bn_real(real_part)
        norm_imag = self.bn_imag(imag_part)
        
        # 恢复形状
        norm_real = norm_real.transpose(1, 2)  # (batch, sequence, features)
        norm_imag = norm_imag.transpose(1, 2)  # (batch, sequence, features)
        
        # 重新组合为复数
        return torch.complex(norm_real, norm_imag)

class resnet(nn.Module):
    '''
    残差连接模块
    Y = W X + \delta X
    '''
    def __init__(self, feature_prex, feature_postx):
        super().__init__()
        self.linear = linear(feature_prex, feature_postx)
    def forward(self, x, deltax):
        x=self.linear(x)
        return x + deltax

class abs(nn.Module):
    '''
    复数取模层
    输入:(batch, sequence, feature)
    '''
    def __init__(self):
        super().__init__()
    def forward(self, x):
        x = torch.abs(x)
        return x

if __name__ == "__main__":
    print("net_complex_basis模块测试")
    tensor_3d = (torch.rand(3, 3, 3)*2-1)+1j*(torch.rand(3, 3, 3)*2-1)
    print("输入tensor:", tensor_3d)
    linear_layer = linear(3, 3, bias=True)
    output = linear_layer(tensor_3d)
    print("线性层输出:", output)
    relu_layer = relu()
    output_relu = relu_layer(tensor_3d)
    print("ReLU层输出:", output_relu)
    tanh_layer = tanh()
    output_tanh = tanh_layer(tensor_3d)
    print("Tanh层输出:", output_tanh)
    sigmoid_layer = sigmoid()
    output_sigmoid = sigmoid_layer(tensor_3d)
    print("Sigmoid层输出:", output_sigmoid)
    bn_layer = NaiveComplexBatchNorm1d(3)
    output_bn = bn_layer(tensor_3d)
    print("批归一化层输出:", output_bn)
    resnet_layer = resnet(3, 3)
    output_resnet = resnet_layer(tensor_3d, tensor_3d)
    print("残差连接层输出:", output_resnet)