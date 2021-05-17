import os
from pathlib import Path
from tqdm import tqdm
from glob import glob
import pandas as pd

for dataset in ["train", "valid", 'test']:
    root = '/Users/rezachi/ML/datasets/aquarium'
    path = Path(root).joinpath(f'{dataset}', '*.jpg')
    all_images = glob(str(path))
    df_path = Path(root).joinpath(f'{dataset}', '_annotations.csv')
    annotation = pd.read_csv(df_path)

    print(f"loading {dataset} data. . .")
    i = 0
    for image in tqdm(all_images):
        img_name = image.split("/")[-1]
        if len(annotation[annotation.filename == img_name]) == 0:
            i += 1
            print(f"removing {image} . . .")
            os.remove(image)
    if i == 0:
        print(f'all {dataset} images were cleaned. Nothing removed')
    else:
        print(f'{i} no images removed from {dataset} data')