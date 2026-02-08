import os
import math
import random
import numpy as np
from PIL import Image
from copy import deepcopy
import cv2
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import albumentations as A
from dataset.transform import *
from torchvision import transforms as T

IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)


class SemiDataset(Dataset):
    def __init__(self, name, root, mode, transform=None, size=None, ignore_value=None, id_path=None, nsample=None):
        self.name = name
        self.root = root
        self.mode = mode
        self.size = size
        self.ignore_value = ignore_value
        self.reduce_zero_label = True if name == 'ade20k' else False
        self.transform = transform

        # if self.mode == 'val':
        #     self.to_tensor = T.Compose([
        #         T.ToTensor(),
        #         T.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
        #     ])

        # else:
        #     self.to_tensor = T.Compose([
        #         T.ToPILImage(),
        #         T.Resize(size),
        #         T.ToTensor(),
        #         T.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
        #     ])



        if mode == 'train_l' or mode == 'train_u':
            with open(id_path, 'r') as f:
                self.ids = f.read().splitlines()
            if mode == 'train_l' and nsample is not None:
                self.ids *= math.ceil(nsample / len(self.ids))
                self.ids = self.ids[:nsample]
        else:
            with open('splits/%s/val_ss_val.txt' % name, 'r') as f:
                self.ids = f.read().splitlines()
            # with open('splits/%s/test.txt' % name, 'r') as f:
                # self.ids = f.read().splitlines()


    def _trainid_to_class(self, label):

        return label

    def tes_class_to_trainid(self, label):
   
        return label

    def _class_to_trainid(self, label):

        return label

    def process_mask(self, mask):
        mask = np.array(mask) - 1
        return Image.fromarray(mask)

    def __len__(self):
        # return len(self.image_list)
        return len(self.ids)

    def __getitem__(self, item):
        id = self.ids[item]
        img = Image.open(os.path.join(self.root, id.split(' ')[0])).convert('RGB')
        mask = Image.open(os.path.join(self.root, id.split(' ')[1]))

        if self.name == 'loveda': 
            mask = self.process_mask(mask)
        

        if self.mode == 'val':
            np_img = np.array(img)
            img, mask = normalize(img, mask)
            return np_img, img, mask, id
            # return img, mask, id

        
        img, mask = resize(img, mask, (0.5, 2.0))
        ignore_value = 254 if self.mode == 'train_u' else self.ignore_value
        img, mask = crop(img, mask, self.size, ignore_value)
        img, mask = hflip(img, mask, p=0.5)
        # img, mask = bflip(img, mask, p=0.5)
        # img, mask = Rotate(img, mask, p=0.5)
        # img, mask = Rotate_90(img, mask, p=0.5)
        # img, mask = Rotate_180(img, mask, p=0.5)
        # img, mask = Rotate_270(img, mask, p=0.5)

        if self.mode == 'train_l':
            return normalize(img, mask)

        img_w, img_s1, img_s2 = deepcopy(img), deepcopy(img), deepcopy(img)

        if random.random() < 0.8:
            img_s1 = transforms.ColorJitter(0.5, 0.5, 0.5, 0.25)(img_s1)
        img_s1 = transforms.RandomGrayscale(p=0.2)(img_s1)
        img_s1 = blur(img_s1, p=0.5)
        cutmix_box = obtain_cutmix_box(img_s1.size[0], p=0.5)

        ignore_mask = Image.fromarray(np.zeros((mask.size[1], mask.size[0])))
        img_s1, ignore_mask = normalize(img_s1, ignore_mask)
        mask = torch.from_numpy(np.array(mask)).long()
        ignore_mask[mask == 254] = self.ignore_value

        return np.array(img_w), normalize(img_w), img_s1, ignore_mask, cutmix_box, mask

    # def __len__(self):
    #     return len(self.ids)
