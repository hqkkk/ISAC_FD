import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

# 参数设置
embed_dim = 64  # 嵌入维度
num_heads = 4   # 注意力头数量
batch_size = 64  # 批次大小
num_epochs = 1000
learning_rate = 0.001

# 生成训练数据
def generate_data(num_samples):
    x = np.random.uniform(-5, 5, num_samples)  # 在[-5, 5]范围内生成随机数
    y = (np.exp(x)-np.sin(x))*np.cos(x)  # 目标函数 e^x
    return torch.tensor(x, dtype=torch.float32).unsqueeze(1), torch.tensor(y, dtype=torch.float32).unsqueeze(1)

# 神经网络架构
class ExpFunctionModel(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(ExpFunctionModel, self).__init__()
        
        # 输入嵌入层 - 将标量输入转换为高维表示
        self.embedding = nn.Linear(1, embed_dim)
        
        # 多头注意力层
        self.attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            batch_first=True
        )
        
        # 输出层
        self.output_layer = nn.Sequential(
            nn.Linear(embed_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
    
    def forward(self, x):
        # 输入形状: (batch_size, 1)
        
        # 嵌入层: 将标量转换为高维向量
        embedded = self.embedding(x)  # 形状: (batch_size, embed_dim)
        
        # 为注意力层添加序列维度 (batch_size, seq_len=1, embed_dim)
        embedded = embedded.unsqueeze(1)
        
        # 多头注意力 - 使用相同的输入作为query, key和value
        attn_output, _ = self.attention(
            query=embedded,
            key=embedded,
            value=embedded,
            need_weights=False
        )  # 形状: (batch_size, 1, embed_dim)
        
        # 移除序列维度
        attn_output = attn_output.squeeze(1)  # 形状: (batch_size, embed_dim)
        
        # 输出层
        output = self.output_layer(attn_output)  # 形状: (batch_size, 1)
        
        return output

# 创建模型
model = ExpFunctionModel(embed_dim, num_heads)
print(model)

# 损失函数和优化器
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# 训练模型
train_losses = []
for epoch in range(num_epochs):
    # 生成训练数据
    x_train, y_train = generate_data(batch_size)
    
    # 前向传播
    outputs = model(x_train)
    loss = criterion(outputs, y_train)
    
    # 反向传播和优化
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    # 记录损失
    train_losses.append(loss.item())
    
    # 每100个epoch打印一次损失
    if (epoch+1) % 100 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.6f}')

# 绘制训练损失
plt.plot(train_losses)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss')
plt.show()

# 测试模型
x_test = torch.linspace(-5, 5, 100).unsqueeze(1)
y_test = (torch.exp(x_test)-torch.sin(x_test))*torch.cos(x_test)
y_pred = model(x_test).detach()

# 绘制结果
plt.figure(figsize=(10, 6))
plt.plot(x_test.numpy(), y_test.numpy(), 'b-', label='True e^x')
plt.plot(x_test.numpy(), y_pred.numpy(), 'r--', label='Predicted')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Fitting e^x with Multihead Attention')
plt.legend()
plt.grid(True)
plt.show()

# 计算测试误差
test_loss = criterion(y_pred, y_test)
print(f'Test Loss: {test_loss.item():.6f}')

# 对照组模型 - 没有注意力层
class ControlModel(nn.Module):
    def __init__(self, embed_dim):
        super(ControlModel, self).__init__()
        
        # 输入嵌入层
        self.embedding = nn.Linear(1, embed_dim)
        
        # 输出层（与实验组相同）
        self.output_layer = nn.Sequential(
            nn.Linear(embed_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
    
    def forward(self, x):
        embedded = self.embedding(x)
        output = self.output_layer(embedded)
        return output

# 训练对照组模型
control_model = ControlModel(embed_dim)
control_optimizer = optim.Adam(control_model.parameters(), lr=learning_rate)

control_losses = []
for epoch in range(num_epochs):
    x_train, y_train = generate_data(batch_size)
    
    outputs = control_model(x_train)
    loss = criterion(outputs, y_train)
    
    control_optimizer.zero_grad()
    loss.backward()
    control_optimizer.step()
    
    control_losses.append(loss.item())
    
    if (epoch+1) % 100 == 0:
        print(f'Control Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.6f}')

# 比较最终损失
plt.figure(figsize=(10, 6))
plt.plot(train_losses, 'b-', label='With Attention')
plt.plot(control_losses, 'r--', label='Without Attention')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss Comparison')
plt.legend()
plt.grid(True)
plt.show()

# 比较测试损失
control_y_pred = control_model(x_test).detach()
control_test_loss = criterion(control_y_pred, y_test)
print(f'Attention Model Test Loss: {test_loss.item():.6f}')
print(f'Control Model Test Loss: {control_test_loss.item():.6f}')