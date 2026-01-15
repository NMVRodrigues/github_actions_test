import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm


def main():
    data = pd.read_parquet(os.path.join("data", "animal_data.parquet"))

    imgs = []
    for img in tqdm(data["img_path"].values):
        with Image.open(img) as img:
            w, h = img.size
            imgs.append([w, h])

    imgs = np.array(imgs)
    print(f"Max image sizes: {np.max(imgs, axis=0)}")
    print(f"Min image sizes: {np.min(imgs, axis=0)}")
    print(f"Avg image sizes: {np.mean(imgs, axis=0)}")


if __name__ == "__main__":
    main()
