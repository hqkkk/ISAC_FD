import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import os

def load_unsupervised_dataset(data_path="unsupervised_dataset"):
    """从文件加载无监督数据集"""
    dataset_path = os.path.join(data_path, "dataset.pt")
    info_path = os.path.join(data_path, "data_info.pt")
    
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"数据集文件不存在: {dataset_path}")
    
    dataset = torch.load(dataset_path)
    data_info = torch.load(info_path)
    
    print(f"加载无监督数据集，大小: {len(dataset)}")
    return dataset, data_info

def unsupervised_loss(outputs, inputs):
    UUPower, DUComMat, INComMat, TAPower = outputs
    UUMat, DUMat, INMat, TAMat, CIMat = inputs
    
    # 示例1: 自编码器重构损失
    # 假设我们希望输出能够重构输入的部分信息
    recon_loss = nn.MSELoss()(UUPower, UUMat.mean(dim=-1, keepdim=True)) + \
                 nn.MSELoss()(TAPower, TAMat.mean(dim=-1, keepdim=True))
    
    # 示例2: 一致性损失 - 鼓励相似输入产生相似输出
    consistency_loss = nn.MSELoss()(DUComMat, INComMat)
    
    # 示例3: 稀疏性约束 - 鼓励输出稀疏
    sparsity_loss = torch.abs(UUPower).mean() + torch.abs(TAPower).mean()
    
    # 组合损失（权重需要根据任务调整）
    total_loss = recon_loss + 0.1 * consistency_loss + 0.01 * sparsity_loss
    
    return total_loss

def unsupervised_train(data_path="unsupervised_dataset", batch_size=32, num_epochs=50, learning_rate=0.001):
    """无监督学习训练函数"""
    
    # 加载数据集
    dataset, data_info = load_unsupervised_dataset(data_path)
    
    # 从数据信息中获取参数
    num_trans = data_info['num_trans']
    num_rece = data_info['num_rece']
    embed_dim = data_info['embed_dim']
    
    # 创建模型
    model = netAll(num_trans, num_rece, num_heads=8, embed_dim=embed_dim, num_update_layers=3)
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    print(f"使用设备: {device}")
    
    # 优化器
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # 创建数据加载器
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # 训练循环
    print("开始无监督训练...")
    train_losses = []
    
    for epoch in range(num_epochs):
        # 训练模式
        model.train()
        epoch_loss = 0.0
        
        for batch in train_loader:
            # 获取输入数据（无监督学习没有标签）
            UUMat_b, DUMat_b, INMat_b, TAMat_b, CIMat_b = batch
            
            # 转移到设备
            UUMat_b = UUMat_b.to(device)
            DUMat_b = DUMat_b.to(device)
            INMat_b = INMat_b.to(device)
            TAMat_b = TAMat_b.to(device)
            CIMat_b = CIMat_b.to(device)
            
            # 前向传播
            optimizer.zero_grad()
            outputs = model(UUMat_b, DUMat_b, INMat_b, TAMat_b, CIMat_b)
            
            # 计算无监督损失
            inputs = (UUMat_b, DUMat_b, INMat_b, TAMat_b, CIMat_b)
            loss = unsupervised_loss(outputs, inputs)
            
            # 反向传播
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_loss)
        
        # 打印进度
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.6f}')
        
        # 保存模型检查点（可选）
        if (epoch + 1) % 10 == 0:
            torch.save(model.state_dict(), f'unsupervised_model_epoch_{epoch+1}.pth')
            print(f"模型已保存: unsupervised_model_epoch_{epoch+1}.pth")
    
    # 保存最终模型
    torch.save(model.state_dict(), 'unsupervised_final_model.pth')
    print("无监督训练完成! 模型已保存为 'unsupervised_final_model.pth'")
    
    return model, train_losses

def visualize_results(model, data_path="unsupervised_dataset", num_examples=5):
    """可视化无监督学习结果"""
    import matplotlib.pyplot as plt
    
    # 加载数据集
    dataset, _ = load_unsupervised_dataset(data_path)
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    
    # 获取几个示例
    examples = []
    for i in range(min(num_examples, len(dataset))):
        example = dataset[i]
        examples.append(example)
    
    # 模型推理
    with torch.no_grad():
        fig, axes = plt.subplots(num_examples, 5, figsize=(15, 3*num_examples))
        if num_examples == 1:
            axes = axes.reshape(1, -1)
        
        for i, example in enumerate(examples):
            UUMat, DUMat, INMat, TAMat, CIMat = example
            
            # 添加批次维度并转移到设备
            inputs = [x.unsqueeze(0).to(device) for x in [UUMat, DUMat, INMat, TAMat, CIMat]]
            
            # 模型推理
            UUPower, DUComMat, INComMat, TAPower = model(*inputs)
            
            # 可视化输入和输出
            # 这里只是示例，需要根据你的数据特点调整可视化方式
            axes[i, 0].imshow(UUMat.cpu().numpy(), aspect='auto', cmap='viridis')
            axes[i, 0].set_title(f'Input UUMat {i+1}')
            
            axes[i, 1].imshow(DUMat.cpu().numpy(), aspect='auto', cmap='viridis')
            axes[i, 1].set_title(f'Input DUMat {i+1}')
            
            axes[i, 2].imshow(UUPower.squeeze().cpu().numpy(), aspect='auto', cmap='viridis')
            axes[i, 2].set_title(f'Output UUPower {i+1}')
            
            axes[i, 3].imshow(DUComMat.squeeze().cpu().numpy(), aspect='auto', cmap='viridis')
            axes[i, 3].set_title(f'Output DUComMat {i+1}')
            
            axes[i, 4].imshow(TAPower.squeeze().cpu().numpy(), aspect='auto', cmap='viridis')
            axes[i, 4].set_title(f'Output TAPower {i+1}')
        
        plt.tight_layout()
        plt.savefig('unsupervised_results.png')
        plt.show()

# 使用示例
if __name__ == "__main__":
    # 进行无监督训练
    model, losses = unsupervised_train("my_unsupervised_data", num_epochs=50)
    
    # 可视化结果
    visualize_results(model, "my_unsupervised_data")