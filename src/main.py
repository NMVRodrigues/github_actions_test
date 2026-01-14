import os
import torch
import pandas as pd
from torch.utils.data import DataLoader
from utils.torch_dataset import AnimalDataset
from sklearn.model_selection import train_test_split
from torchvision.transforms import v2


def main():

    data = os.path.join('data', 'animal_data.parquet')
    data = pd.read_parquet(data)

    train, test = train_test_split(data, test_size=0.1, stratify=data['label'], random_state=42)
    train, val = train_test_split(train, test_size=0.15, stratify=train['label'], random_state=42) 

    transforms = v2.Compose([
        v2.RandomResizedCrop(size=(224, 224), antialias=True),
        v2.RandomHorizontalFlip(p=0.5),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


    train_dataset = AnimalDataset(train)
    val_dataset = AnimalDataset(val)
    test_dataset = AnimalDataset(test)


    train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=64, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=True)

    train_features, train_labels = next(iter(train_dataloader))
    print(f"Feature batch shape: {train_features.size()}")
    print(f"Labels batch shape: {train_labels.size()}")
    img = train_features[0].shape




if __name__ == '__main__':
    main()


