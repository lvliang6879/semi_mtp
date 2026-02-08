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
Image.MAX_IMAGE_PIXELS = None 




def poly2obb_np(polys):
    """
    Convert polygons to oriented bounding boxes (MMRotate style).
    
    Args:
        polys: (N, 8) or (8,) array of [x1,y1,x2,y2,x3,y3,x4,y4]
    
    Returns:
        obbs: (N, 5) or (5,) array of [cx, cy, w, h, theta_rad]
              theta in [-pi/2, pi/2), w >= h, radians
    """
    if polys.ndim == 1:
        single = True
        polys = polys[None, :]
    else:
        single = False

    obbs = []
    for poly in polys:
        # Reshape to (4, 2)
        pts = poly.reshape(4, 2).astype(np.float32)
        
        # Skip degenerate cases
        if cv2.contourArea(pts) < 1.0:
            # Fallback: use axis-aligned bbox
            x_min, y_min = pts.min(axis=0)
            x_max, y_max = pts.max(axis=0)
            cx = (x_min + x_max) / 2
            cy = (y_min + y_max) / 2
            w = max(x_max - x_min, 1.0)
            h = max(y_max - y_min, 1.0)
            theta = 0.0
        else:
            # Get min area rect
            (cx, cy), (w, h), angle_deg = cv2.minAreaRect(pts)
            
            # Ensure w >= h
            if w < h:
                w, h = h, w
                angle_deg += 90.0
            
            # Normalize angle to [-90, 90)
            angle_deg = (angle_deg + 180) % 180 - 180
            if angle_deg < -90:
                angle_deg += 180
            
            theta = math.radians(angle_deg)

        obbs.append([cx, cy, w, h, theta])
    
    obbs = np.array(obbs, dtype=np.float32)
    return obbs[0] if single else obbs



class SemiDataset(Dataset):
    def __init__(self, name, root, mode, transform=None, size=None, ignore_value=None, id_path=None, nsample=None):
        self.name = name
        self.root = root
        self.mode = mode
        self.size = size
        self.ignore_value = ignore_value
        self.transform = transform
        self.CLASS2ID = {
            "large-vehicle": 0,
            "swimming-pool": 1,
            "helicopter": 2,
            "bridge": 3,
            "plane": 4,
            "ship": 5,
            "soccer-ball-field": 6,
            "basketball-court": 7,
            "ground-track-field": 8,
            "small-vehicle": 9,
            "baseball-diamond": 10,
            "tennis-court": 11,
            "roundabout": 12,
            "storage-tank": 13,
            "harbor": 14,
        }
        self.metainfo = {"classes": tuple(self.CLASS2ID.keys()),
                        "dataset_type": "DOTA", "task": "RotatedDetection"}
        # load ids
        if mode in ['train_l', 'train_u']:
            with open(id_path, 'r') as f:
                self.ids = f.read().splitlines()
            if mode == 'train_l' and nsample is not None:
                self.ids = (self.ids * math.ceil(nsample / len(self.ids)))[:nsample]
        else:
            val_path = os.path.join('splits', name, 'val_merge_IRSAMap.txt')
            with open(val_path, 'r') as f:
                self.ids = f.read().splitlines()



    def poly_to_rotated_box(self, poly):
        """
        poly: list or array = [x1,y1, x2,y2, x3,y3, x4,y4]
        return: cx, cy, w, h, angle   (angle in degrees)
        """
        pts = np.array(poly, dtype=np.float32).reshape(4, 2)

        # 最小外接旋转矩形
        rect = cv2.minAreaRect(pts)
        (cx, cy), (w, h), theta = rect

        # OpenCV angle 定义比较特殊，需要处理成常用格式 (horizontal 0 degree)
        # rect angle is in [-90,0), when width < height we rotate 90
        if w < h:
            w, h = h, w
            theta += 90

        return float(cx), float(cy), float(w), float(h), float(theta)


    # def load_dota_label(self, txt_path, class2id):
    #     """
    #     DOTA 格式:
    #         x1 y1 x2 y2 x3 y3 x4 y4 class difficult
    #     """
    #     boxes = []
    #     labels = []

    #     if not os.path.exists(txt_path):
    #         return np.zeros((0, 5), dtype=np.float32), np.zeros((0,), dtype=np.int64)

    #     with open(txt_path, "r") as f:
    #         for line in f.read().splitlines():
    #             parts = line.strip().split()
    #             if len(parts) < 10:
    #                 continue

    #             poly = list(map(float, parts[:8]))
    #             cls_name = parts[8]
    #             difficult = int(parts[9])

    #             # class id
    #             if cls_name not in class2id:
    #                 continue
    #             cid = class2id[cls_name]

    #             # polygon → rotated box
    #             cx, cy, w, h, ang = self.poly_to_rotated_box(poly)

    #             boxes.append([cx, cy, w, h, ang])
    #             labels.append(cid)

    #     if len(boxes) == 0:
    #         return np.zeros((0, 5), dtype=np.float32), np.zeros((0,), dtype=np.int64)

    #     return np.array(boxes, dtype=np.float32), np.array(labels, dtype=np.int64)


    def load_dota_label(self, txt_path, class2id):
        """
        Load DOTA labels and convert to (cx, cy, w, h, theta_rad) with theta in [-pi/2, pi/2)
        """
        boxes = []
        labels = []

        if not os.path.exists(txt_path):
            return np.zeros((0, 5), dtype=np.float32), np.zeros((0,), dtype=np.int64)

        with open(txt_path, 'r') as f:
            lines = f.read().strip().splitlines()
        
        for line in lines:
            parts = line.strip().split()
            if len(parts) < 10:
                continue

            try:
                poly = list(map(float, parts[:8]))
                cls_name = parts[8]
                difficult = int(parts[9])

                if cls_name not in class2id:
                    continue
                cid = class2id[cls_name]

                # Convert to rotated box (radians, [-pi/2, pi/2))
                cx, cy, w, h, theta = poly2obb_np(np.array(poly))

                # Filter invalid boxes
                if w <= 0 or h <= 0:
                    continue

                boxes.append([cx, cy, w, h, theta])
                labels.append(cid)
            except Exception as e:
                continue  # skip malformed lines

        if len(boxes) == 0:
            return np.zeros((0, 5), dtype=np.float32), np.zeros((0,), dtype=np.int64)

        return np.array(boxes, dtype=np.float32), np.array(labels, dtype=np.int64)

    def __len__(self):
        return len(self.ids)

    def _augment_strong(self, img):
        """Strong augmentation for unlabeled images."""
        if random.random() < 0.8:
            img = transforms.ColorJitter(0.5, 0.5, 0.5, 0.25)(img)
        img = transforms.RandomGrayscale(p=0.2)(img)
        img = blur(img, p=0.5)
        return img

    def __getitem__(self, idx):
        id_line = self.ids[idx]
        img_path = os.path.join(self.root, id_line.split(' ')[0])
        img = Image.open(img_path).convert('RGB')
        img_id = idx
        file_name = os.path.splitext(os.path.basename(img_path))[0]
        orig_w, orig_h = img.size
        new_h = self.size
        new_w = self.size
        scale_x = new_w / orig_w
        scale_y = new_h / orig_h

        if self.mode == 'train_u':
            # no labels
            boxes = np.zeros((0, 5), dtype=np.float32)
            labels = np.zeros((0,), dtype=np.int64)
            mask = Image.fromarray(np.zeros((img.size[1], img.size[0]), dtype=np.uint8))
        else:
            # labeled: load DOTA txt
            txt_path = os.path.join(self.root, id_line.split(' ')[1])
            boxes, labels = self.load_dota_label(txt_path, self.CLASS2ID)
            mask_path = os.path.join(self.root, id_line.split(' ')[2])
            mask = Image.open(mask_path)
        # Validation
        if self.mode == 'val':
            # img, boxes = resize_det_val(img, boxes, self.size)
            orig_w, orig_h = img.size
            new_h, new_w = img.size
            scale_x = new_w / orig_w
            scale_y = new_h / orig_h
            img, mask = normalize(img, mask)
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            labels = torch.as_tensor(labels, dtype=torch.int64)
            target = {"boxes": boxes, "labels": labels, "img_id": img_id, 'file_name': file_name,
            "ori_shape": (orig_h, orig_w),  "scale_factor": (scale_x, scale_y), "img_path": img_path}
            return img, mask, target

        # Train augmentations
        img, boxes, mask = resize_(img, boxes, mask, (0.5, 2.0))     # 前面已经写好的 resize_det
        ignore_value = 254 if self.mode == 'train_u' else self.ignore_value
        img, boxes, mask = crop_(img, boxes, mask, self.size, ignore_value)
        img, boxes, mask = hflip_(img, boxes, mask, p=0.5)
        img, boxes, mask = bflip_(img, boxes, mask, p=0.5)
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        target = {"boxes": boxes, "labels": torch.as_tensor(labels, dtype=torch.int64), "img_id": img_id, 'file_name': file_name, 
                 "ori_shape": (orig_h, orig_w),  "scale_factor": (scale_x, scale_y), "img_path": img_path}

        if self.mode == 'train_l':
            img, mask = normalize(img, mask)
            return img, mask, target

        # Unlabeled strong augmentations
        img_w, img_s = img.copy(), img.copy()
        img_s = self._augment_strong(img_s)
        ignore_mask = Image.fromarray(np.zeros((mask.size[1], mask.size[0])))
        img_s, ignore_mask = normalize(img_s, ignore_mask)
        mask = torch.from_numpy(np.array(mask)).long()
        ignore_mask[mask == 254] = self.ignore_value

        target = {
                #  "labels": torch.as_tensor(labels, dtype=torch.int64),
                  "ori_shape": (orig_h, orig_w),  
                  "scale_factor": (scale_x, scale_y), 
                  "img_id": img_id, 
                  'file_name': file_name,
                  "img_path": img_path}
        
        return normalize(img_w), img_s, ignore_mask, target