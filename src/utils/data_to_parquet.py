import os
import pandas as pd
from glob import glob
from tqdm import tqdm


def main():
    data = {'img_path':[], 'label':[]}

    base_folder = glob(os.path.join('data', 'raw-img' ,'*'))

    for animal_folder in tqdm(base_folder, desc='Creating dataframe for the images...'):
        animal = animal_folder.split(os.sep)[-1]
        animal_images = glob(os.path.join(animal_folder, '*.jpeg'))
        for img in animal_images:
            data['img_path'].append(img)
            data['label'] = animal


    df = pd.DataFrame(data)
    df.to_parquet(os.path.join('data', 'animal_data.parquet'))

if __name__ == '__main__':
    main()