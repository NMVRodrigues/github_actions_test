import os
import pandas as pd
from glob import glob
from tqdm import tqdm


def main():
    data = {"img_path": [], "label": []}

    base_folder = glob(os.path.join("data", "raw-img", "*"))

    for animal_folder in tqdm(base_folder, desc="Creating dataframe for the images..."):
        animal = animal_folder.split(os.sep)[-1]
        animal_images = glob(os.path.join(animal_folder, "*.jpeg"))
        for img in animal_images:
            data["img_path"].append(img)
            data["label"] = animal

    df = pd.DataFrame(data)

    label_map = {
        "cane": 0,
        "cavallo": 1,
        "elefante": 2,
        "farfalla": 3,
        "gallina": 4,
        "gatto": 5,
        "mucca": 6,
        "pecora": 7,
        "ragno": 8,
        "scoiattolo": 9,
    }

    df["label_encoded"] = df["label"].apply(lambda x: label_map[x])

    df.to_parquet(os.path.join("data", "animal_data.parquet"))


if __name__ == "__main__":
    main()
