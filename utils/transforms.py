import random
import numpy as np
from torchvision.transforms import functional as F
from .bbox_util import *
import torch

class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target


def check_boxes_validity(boxes):
    for box in boxes:
        if box[0]>box[2]:
            print('invalid bbox . .')
            return 0
        if box[1]>box[3]:
            print('invalid bbox . .')
            return 0

        for pos in box:
            if pos<0:
                print('invalid bbox . .')
                return 0
    return 1      

class RandomHorizontalFlip(object):
    def __init__(self, prob):
        self.prob = prob

    def __call__(self, image, target):

        image_orig,target_orig=image.copy(),target.copy()
        if random.random() < self.prob:
            
            image = np.array(image)
            
            height, width = image.shape[0:2]

            
            image = np.flip(image,axis=1).copy()
            bbox = np.array(target["boxes"])
            bbox[:, [0, 2]] = width - bbox[:, [2, 0]]

            target["boxes"] = bbox

        
        checked = check_boxes_validity(target["boxes"])
        if checked:
            return image, target
        else:
            return image_orig, target_orig


class RandomVerticalFlip(object):
    def __init__(self, prob):
        self.prob = prob

    def __call__(self, image, target):

        image_orig,target_orig=image.copy(),target.copy()
        if random.random() < self.prob:
            
            image = np.array(image)
            
            height, width = image.shape[0:2]

            
            image = np.flip(image,axis=0).copy()
            bbox = np.array(target["boxes"])
            bbox[:, [1, 3]] = height - bbox[:, [3, 1]]
            
            target["boxes"] = bbox
        
        checked = check_boxes_validity(target["boxes"])
        if checked:
            return image, target
        else:
            return image_orig, target_orig


class RandomRotate(object):

    def __init__(self, prob, angle = 10):
        self.angle = angle
        self.prob = prob
        
        if type(self.angle) == tuple:
            assert len(self.angle) == 2, "Invalid range"  
            
        else:
            self.angle = (-self.angle, self.angle)
            
    def __call__(self, img, target):

        image_orig,target_orig=img.copy(),target.copy()
        if random.random() < self.prob:
            angle = random.uniform(*self.angle)

            img = np.array(img)
            bboxes = np.array(target["boxes"])
            
            w,h = img.shape[1], img.shape[0]

           

            cx, cy = w//2, h//2
        
            img = rotate_im(img, angle)
        
            corners = get_corners(bboxes)
        
            corners = np.hstack((corners, bboxes[:,4:]))
        
            corners[:,:8] = rotate_box(corners[:,:8], angle, cx, cy, h, w)
        
            new_bbox = get_enclosing_box(corners)
        
            scale_factor_x = img.shape[1] / w
        
            scale_factor_y = img.shape[0] / h
        
            img = cv2.resize(img, (w,h))
        
            new_bbox[:,:4] /= [scale_factor_x, scale_factor_y, scale_factor_x, scale_factor_y] 
        
            bboxes  = new_bbox
            
            bboxes = clip_box(bboxes, [0,0,w, h], 0.25)
            
            target["boxes"] = bboxes

        checked = check_boxes_validity(target["boxes"])
        if checked:
            return img, target
        else:
            return image_orig, target_orig




class RandomScale(object):

    def __init__(self, prob,scale = 0.2, diff = False):
        self.scale = scale
        self.prob = prob

        
        if type(self.scale) == tuple:
            assert len(self.scale) == 2, "Invalid range"
            assert self.scale[0] > -1, "Scale factor can't be less than -1"
            assert self.scale[1] > -1, "Scale factor can't be less than -1"
        else:
            assert self.scale > 0, "Please input a positive float"
            self.scale = (max(-1, -self.scale), self.scale)
        
        self.diff = diff

        

    def __call__(self, img, target):
    
        
        image_orig,target_orig=img.copy(),target.copy()
        if random.random() < self.prob:

            img = np.array(img)
            bboxes = np.array(target["boxes"])
            

            img_shape = img.shape

            
            
            if self.diff:
                scale_x = random.uniform(*self.scale)
                scale_y = random.uniform(*self.scale)
            else:
                scale_x = random.uniform(*self.scale)
                scale_y = scale_x
                
            resize_scale_x = 1 + scale_x
            resize_scale_y = 1 + scale_y
            
            img=  cv2.resize(img, None, fx = resize_scale_x, fy = resize_scale_y)
            
            bboxes[:,:4] *= [resize_scale_x, resize_scale_y, resize_scale_x, resize_scale_y]
            
            canvas = np.zeros(img_shape, dtype = np.uint8)
            
            y_lim = int(min(resize_scale_y,1)*img_shape[0])
            x_lim = int(min(resize_scale_x,1)*img_shape[1])
            
            
            canvas[:y_lim,:x_lim,:] =  img[:y_lim,:x_lim,:]
            
            img = canvas
            bboxes = clip_box(bboxes, [0,0,1 + img_shape[1], img_shape[0]], 0.25)
            
            target["boxes"] = bboxes

        checked = check_boxes_validity(target["boxes"])
        if checked:
            return img, target
        else:
            return image_orig, target_orig



class ToTensor(object):
    def __call__(self, image, target):

        image=np.array(image)
        target["boxes"] = np.array(target["boxes"])
        target["boxes"] = F.to_tensor(target["boxes"])[0,:,:]
        image = F.to_tensor(image)  
        
        return image, target