import torch
from torch.utils.data import DataLoader
import torchvision

from pathlib import Path
import hydra
from tqdm import tqdm
from PIL import ImageDraw
import numpy as np

from model import GetModel
from dataset import AquaDatasetPredict

from utils.utils import collate_fn

class ModelPredictor(object):
    def __init__(self, cfg):

        # config data
        self.num_classes = cfg.model.num_classes
        self.check_point_path = cfg.model.check_point_path

        self.root = cfg.data.test_root
        self.test_batch = cfg.train.test_batch
        self.img_format = cfg.data.test_img_format
        self.num_workers = cfg.train.num_workers

        self.device = self.get_device()
        self.model = self.make_model(num_classes=self.num_classes, check_point_path=self.check_point_path)
        self.dataloader = self.make_dataset(self.root, self.test_batch)

    def get_device(self):
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        
        print (f'\nDevice is on {device} . . .')
        return device

    def make_model(self, num_classes, check_point_path, model_name='fasterrcnn_resnet50_fpn', all_layers=False):
        model = GetModel(num_classes=num_classes, model_name=model_name, all_layers=all_layers).model
        model.to(self.device)

        checkpoint = torch.load(check_point_path, map_location="cpu")
        model.load_state_dict(checkpoint["model"])

        print('\n    Check point loaded')
        return model

    def make_dataset(self, root, test_batch):
        test_dataset = AquaDatasetPredict(root=root, img_format=self.img_format, resize_image=False)

        test_dataloader = DataLoader(dataset=test_dataset, batch_size=test_batch, shuffle=False, num_workers=self.num_workers, collate_fn=collate_fn)

        return test_dataloader

    @staticmethod
    def draw_boxes_for_one_image(image,labels,scores,boxes, class_map):
        for idx,box in enumerate(boxes):
            if scores[idx] >= 0.5:
                draw = ImageDraw.Draw(image)
                draw.rectangle(box,outline=(255,0,0))
                label = labels[idx]
                text = f'{class_map[label]}-({np.round(scores[idx]*100,1)}%)'
                size = draw.textsize(text)

                draw.rectangle((box[2], box[1], box[2] + size[0], box[1] + size[1]), fill=(0,0,0))
                draw.text([box[2],box[1]],text,fill=(255,255,255))
        return image


    def predict(self):

        class_map = ['bg','starfish', 'shark', 'fish', 'puffin', 'stingray', 'penguin','jellyfish']
        with torch.no_grad():
            self.model.eval()
            # to_save =Path(self.root).parent / 'pred'
            to_save = Path('./pred')
            to_save.mkdir(parents=True,exist_ok=True)
            counter =0
            for image_batch,img_name in  tqdm(iter(self.dataloader)):
                images = list(torchvision.transforms.ToTensor()(img) for img in image_batch)
                images = list(img.to(self.device) for img in images)
                outputs = self.model(images)

                for id,image in enumerate(image_batch):
                    counter +=1
                    out = {k: v.to('cpu').numpy() for k, v in outputs[id].items()}
                    boxes,labels,scores = out['boxes'],out['labels'],out['scores']
                    image = self.draw_boxes_for_one_image(image,labels,scores,boxes, class_map)
                    image.save(to_save / f'./{img_name[id]}.png')

            print(f'prediction done. Images saved')





config_name = './config/config.yaml'
@hydra.main(config_name=config_name)
def predict(cfg):
    ModelPredictor(cfg).predict()




if __name__ == '__main__':
    predict()