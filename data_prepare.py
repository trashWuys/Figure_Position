import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
import numpy as np
from PIL import Image
import os
from torchvision.io import read_image


class ImageDataset(Dataset):
    def __init__(self, images_dir, labels_dir, transform=None):
        self.images_dir = images_dir
        self.labels_dir = labels_dir
        self.transform = transform
        self.images = [os.path.join(images_dir, image) for image in os.listdir(images_dir)]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image_path = self.images[index]
        image = read_image(image_path).float() / 255.0
        if self.transform:
            image = self.transform(image)

        label_path = os.path.join(self.labels_dir, os.path.splitext(os.path.basename(image_path))[0] + '.txt')
        label = torch.from_numpy(np.loadtxt(label_path, dtype=np.float32))
        return image, label