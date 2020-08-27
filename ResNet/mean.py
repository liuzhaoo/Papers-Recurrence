
import torch
import numpy as np
from torchvision import transforms

from torch.utils.data import DataLoader
from dataset import cifa10
abspath = './'

train_dataloader = cifa10(abspath, train=True, transform=transforms.ToTensor())
imgs = torch.stack([img_t for img_t,t in train_dataloader],dim=3)

print(imgs.view(3, -1).mean(dim=1))