import data_prepare
import Model
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
import numpy as np
from PIL import Image
import os
from torchvision.io import read_image

# 定义图像转换（如果需要）
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # 调整图像大小为224x224
    transforms.ToTensor(),  # 将图像转换为PyTorch张量
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 归一化图像
])

# 设置图像和标签的目录
images_dir = 'path/to/images'  # 替换为实际的图像目录路径
labels_dir = 'path/to/labels'  # 替换为实际的标签目录路径

# 创建数据集
dataset = data_prepare.ImageDataset(images_dir=images_dir, labels_dir=labels_dir, transform=transform)

# 创建DataLoader
dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4)  # batch_size:训练样本数

# 迭代DataLoader示例
for images, labels in dataloader:
    # 对图像和标签进行操作
    print(images.shape)  # 示例输出：torch.Size([32, 3, 640, 640])
    print(labels.shape)  # 示例输出：torch.Size([32, <label_dim>])

# 检查是否有CUDA支持的GPU可用，如果有，则使用第一个可用的GPU，如果没有则使用CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = Model.MyModel().to(device)
criterion = Model.custom_loss
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练循环
epochs = 10
for epoch in range(epochs):
    model.train()
    total_loss = 0
    for inputs, targets in dataloader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}, Average Loss: {total_loss / len(dataloader)}")

torch.save(model.state_dict(), 'T_model_weights.pth')
print("Model weights saved to T_model_weights.pth")