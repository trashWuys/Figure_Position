import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
import numpy as np
from PIL import Image
import os
from torchvision.io import read_image


class MyModel(nn.Module):
    #  定义模型和调整输出层
    def __init__(self):
        super(MyModel, self).__init__()
        self.base_model = models.resnet18(pretrained=True)
        self.base_model.fc = nn.Sequential(
            nn.Linear(self.base_model.fc.in_features, 5),
            nn.Sigmoid()  # 此处sigmoid仅用于确保obj, xc, yc的输出在[0,1]范围内
        )

    def forward(self, x):
        x = self.base_model(x)
        obj, xc, yc, du, dv = x[:, 0], x[:, 1], x[:, 2], x[:, 3], x[:, 4]
        obj = torch.sigmoid(obj)  # obj, xc, yc使用sigmoid
        xc = torch.sigmoid(xc)
        yc = torch.sigmoid(yc)
        du = 1.2 * torch.tanh(du)  # du, dv使用tanh并乘以1.2
        dv = 1.2 * torch.tanh(dv)
        return torch.stack([obj, xc, yc, du, dv], dim=1)


def custom_loss(output, target):
    obj_loss = nn.BCELoss()(output[:, 0], target[:, 0])
    mask = target[:, 0] > 0  # obj为0时忽略其他输出的loss
    additional_loss = nn.MSELoss()(output[:, 1:][mask], target[:, 1:][mask]) if mask.any() else 0
    return obj_loss + additional_loss


