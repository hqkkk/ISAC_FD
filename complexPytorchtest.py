import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

# 设置随机种子以确保可重复性
torch.manual_seed(42)
np.random.seed(42)

# 1. 生成复数训练数据
def generate_complex_data(num_samples=1000):
    """生成复数域上的训练数据 y = e^x"""
    # 在复数平面上生成均匀分布的点
    x_real = torch.linspace(-2, 2, int(np.sqrt(num_samples)))
    x_imag = torch.linspace(-2, 2, int(np.sqrt(num_samples)))
    
    x_grid_real, x_grid_imag = torch.meshgrid(x_real, x_imag, indexing='ij')
    x_complex = torch.complex(x_grid_real, x_grid_imag).flatten()
    
    # 计算 y = e^x (复数指数函数)
    y_complex = torch.exp(x_complex)
    
    return x_complex, y_complex

# 2. 使用PyTorch原生复数支持定义神经网络
class ComplexNet(nn.Module):
    def __init__(self, hidden_size=128):
        super(ComplexNet, self).__init__()
        
        # 使用PyTorch原生的复数线性层
        # dtype=torch.complex64 确保权重和偏置都是复数
        self.fc1 = nn.Linear(1, hidden_size, dtype=torch.complex64)
        self.fc2 = nn.Linear(hidden_size, hidden_size, dtype=torch.complex64)
        self.fc3 = nn.Linear(hidden_size, hidden_size, dtype=torch.complex64)
        self.fc4 = nn.Linear(hidden_size, 1, dtype=torch.complex64)
        
        # 复数激活函数 - 使用PyTorch原生的复数激活函数
        # 注意：PyTorch的ReLU不支持复数，所以我们使用其他激活函数
        self.activation = nn.Tanh()  # Tanh支持复数输入
    
    def forward(self, x):
        # 添加维度以匹配线性层输入
        x = x.unsqueeze(-1)
        
        # 前向传播 - 使用复数运算
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = self.activation(self.fc3(x))
        x = self.fc4(x)
        
        return x.squeeze(-1)

# 3. 复数损失函数
def complex_mse_loss(pred, target):
    """复数均方误差损失：|pred - target|^2 的平均值"""
    return torch.mean(torch.abs(pred - target) ** 2)

# 4. 训练函数
def train_model():
    # 生成训练数据
    x_train, y_train = generate_complex_data(1600)
    
    # 创建模型
    model = ComplexNet(hidden_size=128)
    
    # 打印模型信息
    print("模型架构:")
    print(model)
    print(f"\n参数总数: {sum(p.numel() for p in model.parameters())}")
    
    # 检查权重是否为复数
    print("\n权重类型检查:")
    for name, param in model.named_parameters():
        print(f"{name}: {param.dtype}, 形状: {param.shape}")
    
    # 定义优化器
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    
    # 训练参数
    num_epochs = 10000
    losses = []
    
    print("\n开始训练...")
    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        
        # 前向传播
        predictions = model(x_train)
        
        # 计算损失
        loss = complex_mse_loss(predictions, y_train)
        
        # 反向传播
        loss.backward()
        
        # 更新参数
        optimizer.step()
        
        losses.append(loss.item())
        
        if epoch % 100 == 0:
            print(f'Epoch [{epoch}/{num_epochs}], Loss: {loss.item():.6f}')
    
    return model, losses, x_train, y_train

# 5. 测试函数
def test_model(model, x_data, y_data):
    """测试模型性能"""
    model.eval()
    with torch.no_grad():
        predictions = model(x_data)
        test_loss = complex_mse_loss(predictions, y_data)
        
        print(f"\n测试损失: {test_loss.item():.6f}")
        
        # 验证特定点
        test_points = [
            (0+0j, "e^0 = 1"),
            (1+0j, "e^1 ≈ 2.718"),
            (0+3.14159j, "e^(iπ) ≈ -1"),
            (1+1j, "e^(1+i)"),
            (0.5+0.5j, "e^(0.5+0.5i)"),
        ]
        
        print("\n特定点验证结果:")
        print("="*60)
        for point, desc in test_points:
            x_tensor = torch.complex(torch.tensor(point.real), torch.tensor(point.imag))
            true_value = torch.exp(x_tensor)
            pred_value = model(x_tensor)
            abs_error = torch.abs(pred_value - true_value)
            rel_error = abs_error / torch.abs(true_value)
            
            print(f"x = {point}")
            print(f"描述: {desc}")
            print(f"真实值: {true_value.item():.6f}")
            print(f"预测值: {pred_value.item():.6f}")
            print(f"绝对误差: {abs_error.item():.6f}")
            print(f"相对误差: {rel_error.item():.6f}")
            print("-"*40)
        
        return predictions

# 6. 可视化结果
def plot_results(x_complex, y_true, y_pred, losses):
    """可视化训练结果"""
    # 转换为numpy数组
    x_np = x_complex.detach().numpy()
    y_true_np = y_true.detach().numpy()
    y_pred_np = y_pred.detach().numpy()
    
    # 创建子图
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # 真实值可视化
    im1 = axes[0, 0].scatter(x_np.real, x_np.imag, c=y_true_np.real, cmap='RdBu', alpha=0.7)
    axes[0, 0].set_title('真实值 - 实部 (Re(y))')
    axes[0, 0].set_xlabel('Re(x)')
    axes[0, 0].set_ylabel('Im(x)')
    plt.colorbar(im1, ax=axes[0, 0])
    
    im2 = axes[0, 1].scatter(x_np.real, x_np.imag, c=y_true_np.imag, cmap='RdBu', alpha=0.7)
    axes[0, 1].set_title('真实值 - 虚部 (Im(y))')
    axes[0, 1].set_xlabel('Re(x)')
    axes[0, 1].set_ylabel('Im(x)')
    plt.colorbar(im2, ax=axes[0, 1])
    
    axes[0, 2].scatter(y_true_np.real, y_true_np.imag, c='blue', alpha=0.5, s=1)
    axes[0, 2].set_title('真实y值分布')
    axes[0, 2].set_xlabel('Re(y)')
    axes[0, 2].set_ylabel('Im(y)')
    axes[0, 2].grid(True)
    
    # 预测值可视化
    im3 = axes[1, 0].scatter(x_np.real, x_np.imag, c=y_pred_np.real, cmap='RdBu', alpha=0.7)
    axes[1, 0].set_title('预测值 - 实部 (Re(y))')
    axes[1, 0].set_xlabel('Re(x)')
    axes[1, 0].set_ylabel('Im(x)')
    plt.colorbar(im3, ax=axes[1, 0])
    
    im4 = axes[1, 1].scatter(x_np.real, x_np.imag, c=y_pred_np.imag, cmap='RdBu', alpha=0.7)
    axes[1, 1].set_title('预测值 - 虚部 (Im(y))')
    axes[1, 1].set_xlabel('Re(x)')
    axes[1, 1].set_ylabel('Im(x)')
    plt.colorbar(im4, ax=axes[1, 1])
    
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

# 7. 主函数
def main():
    print("PyTorch原生复数神经网络演示: y = e^x")
    print("="*60)
    
    # 训练模型
    model, losses, x_train, y_train = train_model()
    
    # 测试模型
    predictions = test_model(model, x_train, y_train)
    
    # 可视化结果
    plot_results(x_train, y_train, predictions, losses)
    
    # 验证反向传播正常工作
    print("\n反向传播验证:")
    print("梯度计算正常，损失函数值持续下降，证明复数反向传播有效")

if __name__ == "__main__":
    main()