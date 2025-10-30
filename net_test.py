import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

# 设置随机种子以确保结果可重现
torch.manual_seed(42)
np.random.seed(42)

# 生成训练数据 - 在[-2, 2]区间内采样
z_train = torch.linspace(-2, 2, 100).reshape(-1, 1)  # 输入特征，形状为(100, 1)
y_train = torch.exp(z_train)  # 目标值，y = e^z

# 定义神经网络模型
class ExpNet(nn.Module):
    def __init__(self, hidden_size=64):
        super(ExpNet, self).__init__()
        # 使用线性层和激活函数构建网络
        self.network = nn.Sequential(
            nn.Linear(1, hidden_size),  # 输入层到隐藏层
            nn.ReLU(),                  # ReLU激活函数引入非线性
            nn.Linear(hidden_size, hidden_size),  # 隐藏层到隐藏层
            nn.ReLU(),                  # 另一个ReLU激活函数
            nn.Linear(hidden_size, 1)   # 隐藏层到输出层
        )
    
    def forward(self, x):
        return self.network(x)

# 初始化模型、损失函数和优化器
model = ExpNet(hidden_size=64)
criterion = nn.MSELoss()  # 使用均方误差损失函数
optimizer = optim.Adam(model.parameters(), lr=0.01)  # Adam优化器

# 训练模型
losses = []  # 记录训练损失
epochs = 2000

for epoch in range(epochs):
    # 前向传播
    predictions = model(z_train)
    loss = criterion(predictions, y_train)
    
    # 反向传播
    optimizer.zero_grad()  # 清零梯度
    loss.backward()        # 计算梯度
    optimizer.step()      # 更新参数
    
    losses.append(loss.item())
    
    # 每500个epoch打印损失
    if epoch % 500 == 0:
        print(f'Epoch [{epoch}/{epochs}], Loss: {loss.item():.6f}')

# 在测试集上评估模型
z_test = torch.linspace(-2, 2, 50).reshape(-1, 1)
y_true = torch.exp(z_test)
y_pred = model(z_test)

# 计算测试集上的最终损失
test_loss = criterion(y_pred, y_true)
print(f'\nFinal Test Loss: {test_loss.item():.6f}')

# 可视化结果
plt.figure(figsize=(12, 4))

# 子图1: 真实函数与预测结果对比
plt.subplot(1, 2, 1)
plt.plot(z_test.detach().numpy(), y_true.detach().numpy(), 'b-', label='True Function: y = e^z', linewidth=2)
plt.plot(z_test.detach().numpy(), y_pred.detach().numpy(), 'ro', label='Neural Network Predictions', markersize=4)
plt.xlabel('z')
plt.ylabel('y')
plt.title('Function Approximation: y = e^z')
plt.legend()
plt.grid(True, alpha=0.3)

# 子图2: 训练损失曲线
plt.subplot(1, 2, 2)
plt.plot(losses)
plt.yscale('log')  # 使用对数坐标更好地观察损失下降
plt.xlabel('Epoch')
plt.ylabel('Loss (log scale)')
plt.title('Training Loss Curve')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# 打印一些预测样例进行比较
print("\nPrediction Examples:")
print("z\t\tTrue y\t\tPredicted y\t\tError")
print("-" * 55)
for i in range(5):
    z_val = z_test[i*10].item()
    true_val = y_true[i*10].item()
    pred_val = y_pred[i*10].item()
    error = abs(true_val - pred_val)
    print(f"{z_val:.2f}\t\t{true_val:.4f}\t\t{pred_val:.4f}\t\t{error:.4f}")