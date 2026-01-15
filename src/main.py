import os
import torch
import pandas as pd
from torch.utils.data import DataLoader
from utils.torch_dataset import AnimalDataset
from models.vgg16 import VGG16
from sklearn.model_selection import train_test_split
from torchvision.transforms import v2


def main():

    data = os.path.join('data', 'animal_data.parquet')
    data = pd.read_parquet(data)

    train, test = train_test_split(data, test_size=0.1, stratify=data['label'], random_state=42)
    train, val = train_test_split(train, test_size=0.15, stratify=train['label'], random_state=42) 

    transforms = v2.Compose([
        v2.RandomResizedCrop(size=(224, 224), antialias=True), # alternative to simply resizing the image, just make a crop of the desired size
        v2.RandomHorizontalFlip(p=0.5),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    transforms_test = v2.Compose([
        v2.Resize(size=(224, 224)),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


    train_dataset = AnimalDataset(train, transform=transforms)
    val_dataset = AnimalDataset(val, transform=transforms)
    test_dataset = AnimalDataset(test, transform=transforms_test)


    train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=64, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=True)

    train_features, train_labels = next(iter(train_dataloader))
    print(train_features[0].shape)

    model = VGG16()




if __name__ == '__main__':
    main()


