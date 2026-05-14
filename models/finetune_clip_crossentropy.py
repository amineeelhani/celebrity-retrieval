import os
import sys
import torch
import clip
from PIL import Image
from torch import nn                          # CrossEntropyLoss
from torch import optim                       # ottimizzatore
from torch.utils.data import DataLoader       # dataloader
from torchvision import datasets, transforms  # dataset e augmentation

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print(f"Using device: {device}")

model, preprocess = clip.load("ViT-B/32",device)
