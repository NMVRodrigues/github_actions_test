import os
import pandas as pd
import pytest
import torch
from utils.torch_dataset import AnimalDataset
from torchvision.transforms import v2
from torch.utils.data import DataLoader

@pytest.mark.parametrize("batch_size", [8, 16, 32, 64])
def test_different_batch_sizes(batch_size):
    """Test that model works with various batch sizes"""

    data = os.path.join("data", "animal_data.parquet")
    data = pd.read_parquet(data)


    transforms = v2.Compose(
        [
            v2.RandomResizedCrop(
                size=(224, 224), antialias=True
            ),  # alternative to simply resizing the image, just make a crop of the desired size
            v2.RandomHorizontalFlip(p=0.5),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    dataset = AnimalDataset(data, transform=transforms)

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    assert dataloader is not None
    assert len(dataloader) > 0  
    assert dataloader.batch_size == batch_size

