import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import pandas as pd
import pytest
import torch
from src.models.vgg16 import VGG16
from torch.utils.data import DataLoader


@pytest.mark.parametrize("batch_size", [1, 2, 4, 6])
def test_different_batch_sizes(batch_size):
    """Test that model works with various batch sizes"""

    model = VGG16()
    model.eval()  # Set to evaluation mode

    random_input = torch.randn(batch_size, 3, 224, 224)

    with torch.no_grad():
        output = model(random_input)

    # Assertions
    assert output is not None, "Model should return output"
    assert output.shape == (
        batch_size,
        10,
    ), f"Expected shape (4, 10), got {output.shape}"
    assert not torch.isnan(output).any(), "Output should not contain NaN values"
    assert not torch.isinf(output).any(), "Output should not contain Inf values"
