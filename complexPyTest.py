import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from torch.nn.parameter import Parameter

# 1. 定义复数线性层（权重和偏置都是复数）
class ComplexLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super(ComplexLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # 复数权重：实部和虚部都是可训练参数
        self.weight_real = Parameter(torch.randn(out_features, in_features) * 0.1)
        self.weight_imag = Parameter(torch.randn(out_features, in_features) * 0.1)
        
        # 复数偏置
        self.bias_real = Parameter(torch.randn(out_features) * 0.1)
        self.bias_imag = Parameter(torch.randn(out_features) * 0.1)
    
    def forward(self, x):
        # x是复数张量：x = x_real + i*x_imag
        x_real = x.real
        x_imag = x.imag
        
        # 复数矩阵乘法：(a+bi)(c+di) = (ac-bd) + i(ad+bc)
        # 输出实部 = W_real·x_real - W_imag·x_imag + b_real
        output_real = (torch.matmul(x_real, self.weight_real.t()) - 
                      torch.matmul(x_imag, self.weight_imag.t()) + 
                      self.bias_real)
        
        # 输出虚部 = W_real·x_imag + W_imag·x_real + b_imag
        output_imag = (torch.matmul(x_real, self.weight_imag.t()) + 
                      torch.matmul(x_imag, self.weight_real.t()) + 
                      self.bias_imag)
        
        return torch.complex(output_real, output_imag)
    
    def extra_repr(self):
        return f'in_features={self.in_features}, out_features={self.out_features}'

# 2. 定义复数激活函数
class ComplexReLU(nn.Module):
    def forward(self, x):
        # 对复数的模应用ReLU，保持相位不变
        magnitude = torch.abs(x)
        phase = torch.angle(x)
        magnitude_relu = torch.relu(magnitude)
        return magnitude_relu * torch.exp(1j * phase)

# 3. 定义复数神经网络
class ComplexNet(nn.Module):
    def __init__(self, hidden_size=64):
        super(ComplexNet, self).__init__()
        self.network = nn.Sequential(
            ComplexLinear(1, hidden_size),
            ComplexReLU(),
            ComplexLinear(hidden_size, hidden_size),
            ComplexReLU(),
            ComplexLinear(hidden_size, hidden_size),
            ComplexReLU(),
            ComplexLinear(hidden_size, 1)
        )
    
    def forward(self, x):
        # 添加维度以匹配线性层输入
        x = x.unsqueeze(-1)
        x = self.network(x)
        return x.squeeze(-1)

# 4. 生成复数训练数据
def generate_complex_data(num_samples=1000):
    # 在复数平面上生成均匀分布的点
    x_real = torch.linspace(-2, 2, int(np.sqrt(num_samples)))
    x_imag = torch.linspace(-2, 2, int(np.sqrt(num_samples)))
    
    x_grid_real, x_grid_imag = torch.meshgrid(x_real, x_imag, indexing='ij')
    x_complex = torch.complex(x_grid_real, x_grid_imag).flatten()
    
    # 计算 y = e^x (复数指数函数)
    y_complex = torch.exp(x_complex)
    
    return x_complex, y_complex

# 5. 复数损失函数
def complex_mse_loss(pred, target):
    # 复数MSE：|pred - target|^2 的平均值
    return torch.mean(torch.abs(pred - target) ** 2)

# 6. 训练函数
def train_complex_model():
    # 生成数据
    x_complex, y_complex = generate_complex_data(1600)
    
    # 创建模型
    model = ComplexNet(hidden_size=128)
    
    # 定义优化器
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # 训练参数
    num_epochs = 2000
    losses = []
    
    print("开始训练复数神经网络...")
    print(f"模型参数数量: {sum(p.numel() for p in model.parameters())}")
    
    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        
        # 前向传播
        predictions = model(x_complex)
        
        # 计算损失
        loss = complex_mse_loss(predictions, y_complex)
        
        # 反向传播
        loss.backward()
        
        # 检查梯度是否存在（证明反向传播正常工作）
        if epoch == 0:
            print("\n梯度检查:")
            for name, param in model.named_parameters():
                if param.grad is not None:
                    print(f"{name}梯度范数: {param.grad.norm().item():.6f}")
        
        # 更新参数
        optimizer.step()
        
        losses.append(loss.item())
        
        if epoch % 200 == 0:
            print(f'Epoch [{epoch}/{num_epochs}], Loss: {loss.item():.6f}')
    
    return model, losses, x_complex, y_complex

# 7. 测试和评估
def test_model(model, x_complex, y_complex):
    model.eval()
    with torch.no_grad():
        predictions = model(x_complex)
        
        # 计算最终损失
        final_loss = complex_mse_loss(predictions, y_complex)
        print(f"\n最终测试损失: {final_loss.item():.6f}")
        
        # 验证特定点
        test_points = [
            (0+0j, "e^0 = 1"),
            (1+0j, "e^1 ≈ 2.718"),
            (0+3.14159j, "e^(iπ) ≈ -1"),
            (1+1j, "e^(1+i)"),
        ]
        
        print("\n特定点验证结果:")
        print("="*50)
        for point, desc in test_points:
            x_tensor = torch.complex(torch.tensor(point.real), torch.tensor(point.imag))
            true_value = torch.exp(x_tensor)
            pred_value = model(x_tensor)
            error = torch.abs(pred_value - true_value)
            
            print(f"x = {point}")
            print(f"描述: {desc}")
            print(f"真实值: {true_value.item():.6f}")
            print(f"预测值: {pred_value.item():.6f}")
            print(f"绝对误差: {error.item():.6f}")
            print(f"相对误差: {error.item()/torch.abs(true_value).item():.6f}")
            print("-"*30)
        
        return predictions

# 8. 可视化结果
def visualize_results(x_complex, y_complex, predictions, losses):
    # 转换为numpy数组
    x_np = x_complex.detach().numpy()
    y_true_np = y_complex.detach().numpy()
    y_pred_np = predictions.detach().numpy()
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # 真实值可视化
    scatter1 = axes[0, 0].scatter(x_np.real, x_np.imag, c=y_true_np.real, cmap='RdBu', alpha=0.7)
    axes[0, 0].set_title('真实值 - 实部 (Re(y))')
    axes[0, 0].set_xlabel('Re(x)')
    axes[0, 0].set_ylabel('Im(x)')
    plt.colorbar(scatter1, ax=axes[0, 0])
    
    scatter2 = axes[0, 1].scatter(x_np.real, x_np.imag, c=y_true_np.imag, cmap='RdBu', alpha=0.7)
    axes[0, 1].set_title('真实值 - 虚部 (Im(y))')
    axes[0, 1].set_xlabel('Re(x)')
    axes[0, 1].set_ylabel('Im(x)')
    plt.colorbar(scatter2, ax=axes[0, 1])
    
    axes[0, 2].scatter(y_true_np.real, y_true_np.imag, c='blue', alpha=0.5, s=1)
    axes[0, 2].set_title('真实y值分布')
    axes[0, 2].set_xlabel('Re(y)')
    axes[0, 2].set_ylabel('Im(y)')
    axes[0, 2].grid(True)
    
    # 预测值可视化
    scatter3 = axes[1, 0].scatter(x_np.real, x_np.imag, c=y_pred_np.real, cmap='RdBu', alpha=0.7)
    axes[1, 0].set_title('预测值 - 实部 (Re(y))')
    axes[1, 0].set_xlabel('Re(x)')
    axes[1, 0].set_ylabel('Im(x)')
    plt.colorbar(scatter3, ax=axes[1, 0])
    
    scatter4 = axes[1, 1].scatter(x_np.real, x_np.imag, c=y_pred_np.imag, cmap='RdBu', alpha=0.7)
    axes[1, 1].set_title('预测值 - 虚部 (Im(y))')
    axes[1, 1].set_xlabel('Re(x)')
    axes[1, 1].set_ylabel('Im(x)')
    plt.colorbar(scatter4, ax=axes[1, 1])
    
    axes[1, 2].scatter(y_pred_np.real, y_pred_np.imag, c='red', alpha=0.5, s=1)
    axes[1, 2].set_title('预测y值分布')
    axes[1, 2].set_xlabel('Re(y)')
    axes[1, 2].set_ylabel('Im(y)')
    axes[1, 2].grid(True)
    
    plt.tight_layout()
    plt.show()
    
    # 绘制损失曲线
    plt.figure(figsize=(10, 5))
    plt.semilogy(losses)
    plt.title('复数神经网络训练损失曲线')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (log scale)')
    plt.grid(True)
    plt.show()

# 9. 主函数
def main():
    print("复数神经网络演示: y = e^x")
    print("="*50)
    
    # 训练模型
    model, losses, x_complex, y_complex = train_complex_model()
    
    # 测试模型
    predictions = test_model(model, x_complex, y_complex)
    
    # 可视化结果
    visualize_results(x_complex, y_complex, predictions, losses)
    
    # 验证复数线性层的权重确实是复数
    print("\n验证线性层权重为复数:")
    for name, module in model.named_modules():
        if isinstance(module, ComplexLinear):
            weight_complex = torch.complex(module.weight_real, module.weight_imag)
            print(f"{name}.weight 形状: {weight_complex.shape}")
            print(f"权重类型: 复数 (实部和虚部)")
            print(f"权重示例: {weight_complex[0, 0].item():.6f}")
            break

if __name__ == "__main__":
    main()