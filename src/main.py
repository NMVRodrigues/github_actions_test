import os
import torch
import pandas as pd
from tqdm import tqdm
from torch.utils.data import DataLoader
from utils.torch_dataset import AnimalDataset
from models.vgg16 import VGG16
from sklearn.model_selection import train_test_split
from torchvision.transforms import v2


def main():

    LEARNING_RATE = 1e-4
    EPOCHS = 2
    DEVICE = 'cuda'

    data = os.path.join('data', 'animal_data.parquet')
    data = pd.read_parquet(data)

    train, test = train_test_split(data, test_size=0.1, stratify=data['label_encoded'], random_state=42)
    train, val = train_test_split(train, test_size=0.15, stratify=train['label_encoded'], random_state=42) 

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


    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=True)

    total_steps = len(train_dataloader)

    train_features, train_labels = next(iter(train_dataloader))
    print(train_features[0].shape)

    model = VGG16().to(DEVICE)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay = 0.005)  

    for epoch in range(EPOCHS):
    
        loop = tqdm(train_dataloader, leave=False, desc=f"Epoch {epoch+1}/{EPOCHS}")
        
        for images, labels in loop: 
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # This updates continuously on the same line
            loop.set_postfix(loss=f"{loss.item():.4f}")

        with torch.no_grad():
            correct = 0
            total = 0
            for images, labels in val_dataloader:
                images = images.to(DEVICE)
                labels = labels.to(DEVICE)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        val_acc = 100 * correct / total
        print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {loss.item():.4f} | Val Acc: {val_acc:.2f}%")





if __name__ == '__main__':
    main()


