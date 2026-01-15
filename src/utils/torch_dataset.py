import os
import torch
import pandas as pd
from torch.utils.data import Dataset
from torchvision.io import decode_image


class AnimalDataset(Dataset):
    def __init__(self, annotations_file, transform=None, target_transform=None):
        self.data = annotations_file
        self.img_labels = self.data["label_encoded"].values
        self.img_dir = self.data["img_path"].values
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = self.img_dir[idx]
        image = decode_image(img_path)
        label = torch.tensor(self.img_labels[idx])
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label
