import random
import numpy as np
from pathlib import Path
import hydra


import torch
from torch.utils.data import DataLoader

import utils.transforms as T
from utils.utils import collate_fn, get_mf1, save_on_master
from dataset import AquaDataset
from model import GetModel
from utils.engine import train_one_epoch, evaluate

from torch.utils.tensorboard import SummaryWriter



def get_transform(train):
    transforms = []

    if train:
        transforms.append(T.RandomHorizontalFlip(prob=0.0))
        transforms.append(T.RandomRotate(prob=0.0,angle=10))
        transforms.append(T.RandomScale(prob=0.0,scale=0.2))

    transforms.append(T.ToTensor())

    return T.Compose(transforms)

class ModelTrainer(object):
    def __init__(self, cfg):
        self.seed=cfg.train.seed
        random.seed(self.seed) 
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)

        # config data
        self.num_classes = cfg.model.num_classes
        self.model_name = cfg.model.model_name
        self.lr = cfg.train.lr
        self.momentum = cfg.train.momentum
        self.step_size = cfg.train.step_size
        self.gamma = cfg.train.gamma
        self.root = cfg.data.root
        self.check_point = cfg.model.check_point_path
        self.train_batch = cfg.train.train_batch
        self.val_batch = cfg.train.val_batch
        self.num_epochs = cfg.train.epochs
        self.num_workers = cfg.train.num_workers
        self.all_layers = cfg.train.all_layers

        self.device = self.get_device()
        self.model = self.make_model(num_classes=self.num_classes, model_name=self.model_name, all_layers=self.all_layers)
        self.optimizer, self.lr_scheduler = self.make_optimizer(lr=self.lr, momentum=self.momentum, step_size=self.step_size, gamma=self.gamma)
        self.train_dataloader, self.val_dataloader = self.make_dataset(root=self.root, train_batch=self.train_batch, val_batch=self.val_batch)

    def get_device(self):
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        
        print (f'\ntraining on {device} . . .')
        return device

    def make_model(self, num_classes, model_name, all_layers):
        model = GetModel(num_classes=num_classes, model_name=model_name, all_layers=all_layers).model
        model.to(self.device)

        print('\n    Total params: %.2fM No' % (sum(p.numel() for p in model.parameters())/1000000.0))
        print('    Total trainable params: %.0f No' % (sum(p.numel() for p in model.parameters() if p.requires_grad)))
        return model

    def make_optimizer(self, lr, momentum, step_size, gamma):
        optimizer = torch.optim.SGD(params=self.model.parameters(), lr=lr, momentum=momentum, weight_decay=0.0005)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=step_size, gamma=gamma)

        return optimizer, lr_scheduler

    def make_dataset(self, root, train_batch, val_batch):
        train_dataset = AquaDataset(root=root, dataset='train', transforms=get_transform(train=True))
        val_dataset = AquaDataset(root=root, dataset='valid', transforms=get_transform(train=False))

        train_dataloader = DataLoader(dataset=train_dataset, batch_size=train_batch, shuffle=True, num_workers=self.num_workers, collate_fn=collate_fn)
        val_dataloader = DataLoader(dataset=val_dataset, batch_size=val_batch, shuffle=False, num_workers=self.num_workers, collate_fn=collate_fn)

        return train_dataloader, val_dataloader

    def train(self):
        start_epoch = 1
        to_save=Path('./checkpoint')
        to_save.mkdir(parents=True, exist_ok=True)

        best_F1 = 0
        counter=0

        writer = SummaryWriter(log_dir='./tb_log')

        for epoch in range(start_epoch, self.num_epochs+start_epoch):
            print(f'\nEpoch: [{epoch} | {self.num_epochs}] ')

            loss_dict_reduced, counter= train_one_epoch(self.model, self.optimizer, self.train_dataloader, self.device, epoch,log_dir='./tb_log', counter=counter)
            self.lr_scheduler.step()

            for loss in loss_dict_reduced.keys():
                writer.add_scalar(f'loss/{loss}', loss_dict_reduced[loss].item(), epoch)

            loss_value = sum(loss for loss in loss_dict_reduced.values()).item()
            writer.add_scalar(f'loss/total_loss', loss_value, epoch)
            

            result= evaluate(self.model, self.val_dataloader, self.device)
            stats = result.coco_eval["bbox"].stats
            mF1 = get_mf1(stats)

            print(f"Mean F1 score = {mF1}")
            writer.add_scalar(f'mF1', mF1, epoch)
            writer.flush()

            if mF1 > best_F1:
                best_F1 = mF1

                print(f"Saving the model . . ")
                save_on_master(
                    {
                        "model": self.model.state_dict(),
                        "optimizer": self.optimizer.state_dict(),
                        "lr_scheduler": self.lr_scheduler.state_dict(),
                        "epoch": epoch,
                    },
                    to_save / f"mF1-{np.round(mF1,2)}.pth",
                )
        writer.close()


config_name = './config/config.yaml'
@hydra.main(config_name=config_name)
def run(cfg):
    ModelTrainer(cfg).train()
   






if __name__ == '__main__':
    run()
    

    