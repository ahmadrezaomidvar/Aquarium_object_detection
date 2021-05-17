import pandas as pd
from pathlib import Path
from glob import glob
from PIL import Image
import numpy as np

import torch
from torch.utils.data import Dataset


class AquaDataset(Dataset):
    def __init__(self, root, dataset, transforms=None):
        super(AquaDataset, self).__init__()
        self.root = root
        self.dataset = dataset
        path = Path(self.root).joinpath(f'{self.dataset}', '*.jpg')
        self.imgs = glob(str(path))
        df_path = Path(self.root).joinpath(f'{self.dataset}', '_annotations.csv')
        self.annotation = pd.read_csv(df_path)
        self.annotation['class'] = (self.annotation['class'].factorize()[0]+1)
        self.transforms = transforms

    def __getitem__(self, index):
        img_path = self.imgs[index]
        img = Image.open(img_path).convert('RGB')
        w, h = img.size
        img_name = img_path.split('/')[-1]
        annotation = self.annotation[self.annotation.filename == img_name]
        obj_ids = annotation['class'].values

        boxes = []
        for idx , ids in enumerate(obj_ids):
            xmin = annotation['xmin'].iloc[idx]
            xmax = annotation['xmax'].iloc[idx]
            ymin = annotation['ymin'].iloc[idx]
            ymax = annotation['ymax'].iloc[idx]

            boxes.append([max(xmin, 0), max(ymin, 0), min(xmax, w), min(ymax, h)])

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(obj_ids, dtype=torch.int64)
        image_id = torch.tensor([index])
        area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])

        iscrowd = torch.zeros((len(obj_ids),), dtype=torch.int64)

        target = {}
        target['boxes'] = boxes
        target['labels'] = labels
        target['image_id'] = image_id
        target['area'] = area
        target['iscrowd'] = iscrowd

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.imgs)


class AquaDatasetPredict(Dataset):
    def __init__(self, root, transforms=None,img_format = 'jpg',resize_image=False):
       
        self.root = root
        path = Path(root) / f"*.{img_format}"
        self.imgs = glob(str(path))
        self.resize_image = resize_image

    def __getitem__(self, index):
        
        img_path = self.imgs[index]
        img_name = img_path.split('/')[-1]
        img = Image.open(img_path).convert("RGB")
        if self.resize_image:
            img = img.resize((400,600))


        return img, img_name

    def __len__(self):
        return len(self.imgs)



if __name__ == '__main__':
    print('')
