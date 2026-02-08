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

# ========== 你的 SAR 特定配置 ==========
num_classes = 6
# num_classes = 4
XT_COLORMAP = [[255, 255, 0], [0, 0, 255], [0, 255, 0], 
               [255, 0, 0], [0, 255, 255], [255, 255, 255]]

# XT_COLORMAP = [[255, 255, 0], [0, 0, 255], [0, 255, 0], [0, 255, 255]]

# XT_MEAN = np.array([134.19, 181.63, 179.26, 141.37])
# XT_STD  = np.array([141.34, 166.50, 164.96, 138.58])

XT_MEAN = np.array([141.37, 141.37, 141.37])
XT_STD  = np.array([138.58, 138.58, 138.58])

# 构建 colormap -> label 查找表
colormap2label = np.full(256 ** 3, 255, dtype=np.uint8)
for i, cm in enumerate(XT_COLORMAP):
    idx = (cm[0] * 256 + cm[1]) * 256 + cm[2]
    colormap2label[idx] = i


def Color2Index(ColorLabel):
    data = ColorLabel.astype(np.int32)
    idx = (data[:, :, 0] * 256 + data[:, :, 1]) * 256 + data[:, :, 2]
    IndexMap = colormap2label[idx]  # 已经是 0～3 或 255
    return IndexMap.astype(np.uint8)

def sar_normalize(sar):
    normalized_sar = sar.astype(np.float32)
    normalized_sar = (normalized_sar - XT_MEAN) / XT_STD
    return normalized_sar.astype(np.float32)

class SemiDataset(Dataset):
    def __init__(self, name, root, mode, transform=None, size=None, ignore_value=None, id_path=None, nsample=None):
        self.name = name
        self.root = root
        self.mode = mode
        self.size = size
        self.ignore_value = ignore_value
        self.reduce_zero_label = True if name == 'ade20k' else False
        self.transform = transform


        if mode == 'train_l' or mode == 'train_u':
            with open(id_path, 'r') as f:
                self.ids = f.read().splitlines()
            if mode == 'train_l' and nsample is not None:
                self.ids *= math.ceil(nsample / len(self.ids))
                self.ids = self.ids[:nsample]
        else:
            with open('splits/%s/val.txt' % name, 'r') as f:
                self.ids = f.read().splitlines()
            # with open('splits/%s/test.txt' % name, 'r') as f:
            #     self.ids = f.read().splitlines()


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
        return len(self.ids)

    def __getitem__(self, item):
        id = self.ids[item]
        img = Image.open(os.path.join(self.root, id.split(' ')[0])).convert('L')
        img = Image.merge('RGB', (img, img, img)) 
        mask = np.array(Image.open(os.path.join(self.root, id.split(' ')[1])))
        mask = Image.fromarray(Color2Index(mask))

        img, mask = resize(img, mask, (0.5, 2.0))
        ignore_value = 254 if self.mode == 'train_u' else self.ignore_value
        img, mask = crop(img, mask, self.size, ignore_value)

        img, mask = hflip(img, mask, p=0.5)
        img, mask = bflip(img, mask, p=0.5)
        img, mask = Rotate_90(img, mask, p=0.5)
        # img = np.array(img)
        # img = sar_normalize(img)
        # img = np.transpose(img, (2, 0, 1))
        # mask = torch.from_numpy(np.array(mask)).long()
        return normalize(img, mask)



class ValDataset(Dataset):
    def __init__(self, name, root, mode, size=None,  ignore_value=None, id_path=None, nsample=None):
        self.name = name
        self.root = root
        self.mode = mode
        self.size = size
        self.ignore_value = ignore_value
        self.reduce_zero_label = True if name == 'ade20k' else False

        if mode == 'train_l' or mode == 'train_u':
            with open(id_path, 'r') as f:
                self.ids = f.read().splitlines()
            if mode == 'train_l' and nsample is not None:
                self.ids *= math.ceil(nsample / len(self.ids))
                self.ids = self.ids[:nsample]
        else:
            with open('splits/%s/val.txt' % name, 'r') as f:
                self.ids = f.read().splitlines()

    def __getitem__(self, item):
        id = self.ids[item]
        img = Image.open(os.path.join(self.root, id.split(' ')[0])).convert('L')
        img = Image.merge('RGB', (img, img, img)) 
        img = np.array(img)
        mask = np.array(Image.open(os.path.join(self.root, id.split(' ')[1])))
    

        if self.mode == 'val':
            # mask = Color2Index(mask)
            mask = Image.fromarray(Color2Index(mask))
            # img = sar_normalize(img)
            # img = np.transpose(img, (2, 0, 1))
            # mask = torch.from_numpy(np.array(mask)).long()
            # return img, mask, id
            return normalize(img, mask)

    def __len__(self):
        return len(self.ids)

