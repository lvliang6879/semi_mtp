import random
import numpy as np
from PIL import Image, ImageOps, ImageFilter
import math
import torch
from torchvision import transforms


def crop(img, mask, size, ignore_value=255):
    w, h = img.size
    padw = size - w if w < size else 0
    padh = size - h if h < size else 0
    img = ImageOps.expand(img, border=(0, 0, padw, padh), fill=0)
    mask = ImageOps.expand(mask, border=(0, 0, padw, padh), fill=ignore_value)

    w, h = img.size
    x = random.randint(0, w - size)
    y = random.randint(0, h - size)
    img = img.crop((x, y, x + size, y + size))
    mask = mask.crop((x, y, x + size, y + size))

    return img, mask


def hflip(img, mask, p=0.5):
    if random.random() < p:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
        mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
    return img, mask



def bflip(img, mask, p=0.5):
    if random.random() < p:
        img = img.transpose(Image.FLIP_TOP_BOTTOM)
        mask = mask.transpose(Image.FLIP_TOP_BOTTOM)
    return img, mask

def Rotate(img, mask, p=0.5):  # [-30, 30]
    if random.random() < p:
        v = random.randint(-30, 30)
        return img.rotate(v), mask.rotate(v)
    else:
        return img, mask

def Rotate_90(img, mask, p=0.5):
    if random.random() < p:
        v = 90
        return img.rotate(v), mask.rotate(v)
    else:
        return img, mask

def Rotate_180(img, mask, p=0.5):
    if random.random() < p:
        v = 180
        return img.rotate(v), mask.rotate(v)
    else:
        return img, mask

def Rotate_270(img, mask, p=0.5):
    if random.random() < p:
        v = 270
        return img.rotate(v), mask.rotate(v)
    else:
        return img, mask


def normalize(img, mask=None):
    img = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])(img)
    if mask is not None:
        mask = torch.from_numpy(np.array(mask)).long()
        return img, mask
    return img



def resize(img, mask, ratio_range):
    w, h = img.size
    long_side = random.randint(int(max(h, w) * ratio_range[0]), int(max(h, w) * ratio_range[1]))

    if h > w:
        oh = long_side
        ow = int(1.0 * w * long_side / h + 0.5)
    else:
        ow = long_side
        oh = int(1.0 * h * long_side / w + 0.5)

    img = img.resize((ow, oh), Image.BILINEAR)
    mask = mask.resize((ow, oh), Image.NEAREST)
    return img, mask


def blur(img, p=0.5):
    if random.random() < p:
        sigma = np.random.uniform(0.1, 2.0)
        img = img.filter(ImageFilter.GaussianBlur(radius=sigma))
    return img


def obtain_cutmix_box(img_size, p=0.5, size_min=0.02, size_max=0.4, ratio_1=0.3, ratio_2=1/0.3):
    mask = torch.zeros(img_size, img_size)
    if random.random() > p:
        return mask

    size = np.random.uniform(size_min, size_max) * img_size * img_size
    while True:
        ratio = np.random.uniform(ratio_1, ratio_2)
        cutmix_w = int(np.sqrt(size / ratio))
        cutmix_h = int(np.sqrt(size * ratio))
        x = np.random.randint(0, img_size)
        y = np.random.randint(0, img_size)

        if x + cutmix_w <= img_size and y + cutmix_h <= img_size:
            break

    mask[y:y + cutmix_h, x:x + cutmix_w] = 1

    return mask


########################## Detection Augmentations ##########################


# def resize_det_val(img, boxes, out_size):
#     """
#     img: PIL Image
#     boxes: numpy array [N,5] (cx,cy,w,h,angle)
#     out_size: int or (H,W)
#     """
#     if isinstance(out_size, int):
#         out_h, out_w = out_size, out_size
#     else:
#         out_h, out_w = out_size

#     w, h = img.size
#     scale_x = out_w / w
#     scale_y = out_h / h

#     # resize image
#     img = img.resize((out_w, out_h), Image.BILINEAR)

#     # resize boxes
#     if boxes is not None and len(boxes) > 0:
#         boxes = boxes.copy()
#         boxes[:, 0] *= scale_x   # cx
#         boxes[:, 1] *= scale_y   # cy
#         boxes[:, 2] *= scale_x   # w
#         boxes[:, 3] *= scale_y   # h
#         # angle 不变

#     return img, boxes


# def resize_det(img, boxes, ratio_range):
#     """
#     Resize for rotated object detection
#     img: PIL RGB image
#     boxes: numpy array (N,5) [cx, cy, w, h, angle]
#     ratio_range: (min_ratio, max_ratio)
#     """

#     w, h = img.size

#     # ---- 随机选择目标长边 ----
#     long_side_new = random.randint(
#         int(max(h, w) * ratio_range[0]),
#         int(max(h, w) * ratio_range[1])
#     )

#     # ---- 等比例缩放 ----
#     if h > w:
#         new_h = long_side_new
#         new_w = int(w * long_side_new / h + 0.5)
#     else:
#         new_w = long_side_new
#         new_h = int(h * long_side_new / w + 0.5)

#     # ---- 缩放比例 ----
#     scale_x = new_w / w
#     scale_y = new_h / h

#     # ---- resize image ----
#     img = img.resize((new_w, new_h), Image.BILINEAR)

#     # ---- resize rotated boxes ----
#     if boxes is not None and len(boxes) > 0:
#         boxes_new = boxes.copy()
#         # cx 和 cy 按各自方向缩放
#         boxes_new[:, 0] = boxes[:, 0] * scale_x   # cx
#         boxes_new[:, 1] = boxes[:, 1] * scale_y   # cy
#         # w 和 h 按对应方向缩放
#         boxes_new[:, 2] = boxes[:, 2] * scale_x   # w
#         boxes_new[:, 3] = boxes[:, 3] * scale_y   # h
#         # angle 不变
#     else:
#         boxes_new = boxes

#     return img, boxes_new



# def crop_det(img, boxes, size):
#     """
#     针对旋转检测的随机裁剪:
#       - 不足 size 时右下 pad
#       - 随机裁一个 size x size
#       - boxes 的中心减去裁剪偏移 (x, y)
#       - 只保留中心仍在裁剪区域内的 box
#     """
#     w, h = img.size
#     padw = size - w if w < size else 0
#     padh = size - h if h < size else 0

#     if padw > 0 or padh > 0:
#         # 只在右和下 pad，坐标系不变
#         img = ImageOps.expand(img, border=(0, 0, padw, padh), fill=0)
#         w, h = img.size

#     # 随机裁剪左上角坐标
#     x = random.randint(0, w - size)
#     y = random.randint(0, h - size)

#     img = img.crop((x, y, x + size, y + size))

#     if boxes is None or len(boxes) == 0:
#         return img, boxes

#     boxes_new = boxes.copy()
#     # 中心坐标平移
#     boxes_new[:, 0] = boxes[:, 0] - x  # cx
#     boxes_new[:, 1] = boxes[:, 1] - y  # cy

#     # 保留中心落在新图中的 box
#     keep = (
#         (boxes_new[:, 0] > 0) & (boxes_new[:, 0] < size) &
#         (boxes_new[:, 1] > 0) & (boxes_new[:, 1] < size)
#     )
#     boxes_new = boxes_new[keep]

#     return img, boxes_new



# def _normalize_angle_deg(a):
#     # 归一化到 [-180, 180)
#     return ((a + 180) % 360) - 180

# def hflip_det(img, boxes, p=0.5):
#     """
#     水平翻转:
#       x' = w - x
#       angle' = -angle  (再归一化)
#     """
#     if random.random() >= p:
#         return img, boxes

#     img = img.transpose(Image.FLIP_LEFT_RIGHT)
#     if boxes is None or len(boxes) == 0:
#         return img, boxes

#     w, _ = img.size
#     boxes_new = boxes.copy()
#     boxes_new[:, 0] = w - boxes[:, 0]        # cx
#     boxes_new[:, 4] = -boxes[:, 4]           # angle

#     boxes_new[:, 4] = _normalize_angle_deg(boxes_new[:, 4])

#     return img, boxes_new

# def bflip_det(img, boxes, p=0.5):
#     """
#     垂直翻转:
#       y' = h - y
#       angle' = -angle  (再归一化)
#     """
#     if random.random() >= p:
#         return img, boxes

#     img = img.transpose(Image.FLIP_TOP_BOTTOM)
#     if boxes is None or len(boxes) == 0:
#         return img, boxes

#     _, h = img.size
#     boxes_new = boxes.copy()
#     boxes_new[:, 1] = h - boxes[:, 1]        # cy
#     boxes_new[:, 4] = -boxes[:, 4]           # angle

#     boxes_new[:, 4] = _normalize_angle_deg(boxes_new[:, 4])

#     return img, boxes_new

def resize_det_val(img, boxes, out_size):
    """
    Resize for validation in rotated object detection.
    
    Args:
        img: PIL Image
        boxes: numpy array [N,5] (cx,cy,w,h,angle)
        out_size: int or (H,W) for output size
    
    Returns:
        img: resized and padded PIL Image
        boxes: resized boxes with same format
    """
    if isinstance(out_size, int):
        out_h, out_w = out_size, out_size
    else:
        out_h, out_w = out_size

    w, h = img.size
    scale = min(out_w / w, out_h / h)

    # Calculate new dimensions while keeping aspect ratio
    new_w = int(w * scale)
    new_h = int(h * scale)

    # Resize image
    img = img.resize((new_w, new_h), Image.BILINEAR)

    # Pad to target size
    delta_w = out_w - new_w
    delta_h = out_h - new_h
    padding = (0, 0, delta_w, delta_h)  # left, top, right, bottom
    img = ImageOps.expand(img, padding, fill=0)

    # Resize boxes
    if boxes is not None and len(boxes) > 0:
        boxes = boxes.copy()
        boxes[:, 0] *= scale   # cx
        boxes[:, 1] *= scale   # cy
        boxes[:, 2] *= scale   # w
        boxes[:, 3] *= scale   # h
        # angle 不变

    return img, boxes

def normalize_det(img):
    """
    图像标准化，返回 tensor [3,H,W]
    boxes 不在这里处理
    """
    t = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])
    return t(img)

def _normalize_angle_rad(theta):
    """
    Normalize angle to [-pi/2, pi/2)
    Used in MMRotate and Rotated R-CNN.
    """
    theta = theta % (2 * math.pi)  # [0, 2π)
    if theta > math.pi:
        theta -= 2 * math.pi       # [-π, π)
    if theta > math.pi / 2:
        theta -= math.pi           # [-π/2, π/2)
    elif theta < -math.pi / 2:
        theta += math.pi
    return theta


def resize_det(img, boxes, ratio_range):
    """
    Resize image and boxes by scaling the long side randomly.
    boxes: (N, 5) [cx, cy, w, h, theta_rad]
    """
    w, h = img.size
    long_side = max(h, w)

    # Random long side within range
    new_long = random.randint(
        int(long_side * ratio_range[0]),
        int(long_side * ratio_range[1])
    )

    # Keep aspect ratio
    if h > w:
        new_h = new_long
        new_w = int(w * new_long / h + 0.5)
    else:
        new_w = new_long
        new_h = int(h * new_long / w + 0.5)

    scale_x = new_w / w
    scale_y = new_h / h

    img = img.resize((new_w, new_h), Image.BILINEAR)

    if boxes is not None and len(boxes) > 0:
        boxes = boxes.copy()
        boxes[:, 0] *= scale_x  # cx
        boxes[:, 1] *= scale_y  # cy
        boxes[:, 2] *= scale_x  # w
        boxes[:, 3] *= scale_y  # h
        # theta unchanged

    return img, boxes


def crop_det(img, boxes, size):
    """
    Random crop to (size, size). Pad if needed.
    Keep boxes whose **at least one corner is inside** the crop.
    """
    w, h = img.size
    padw = max(0, size - w)
    padh = max(0, size - h)

    if padw > 0 or padh > 0:
        img = ImageOps.expand(img, border=(0, 0, padw, padh), fill=0)
        w, h = img.size

    x = random.randint(0, w - size)
    y = random.randint(0, h - size)
    img = img.crop((x, y, x + size, y + size))

    if boxes is None or len(boxes) == 0:
        return img, boxes

    boxes = boxes.copy()
    boxes[:, 0] -= x  # cx
    boxes[:, 1] -= y  # cy

    # --- Better box filtering: check if any corner is in crop ---
    cx, cy, w, h, theta = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3], boxes[:, 4]
    cos_a = np.cos(theta)
    sin_a = np.sin(theta)

    # Compute 4 corners relative to center
    dx = w / 2
    dy = h / 2
    corners_x = np.stack([
        cx + dx * cos_a + dy * sin_a,
        cx - dx * cos_a + dy * sin_a,
        cx - dx * cos_a - dy * sin_a,
        cx + dx * cos_a - dy * sin_a,
    ], axis=1)  # (N, 4)
    corners_y = np.stack([
        cy - dx * sin_a + dy * cos_a,
        cy + dx * sin_a + dy * cos_a,
        cy + dx * sin_a - dy * cos_a,
        cy - dx * sin_a - dy * cos_a,
    ], axis=1)

    # Check if any corner is inside [0, size)
    in_crop = (
        (corners_x >= 0) & (corners_x < size) &
        (corners_y >= 0) & (corners_y < size)
    ).any(axis=1)

    boxes = boxes[in_crop]
    return img, boxes



def hflip_det(img, boxes, p=0.5):
    """Horizontal flip: x' = W - x, θ' = -θ"""
    if random.random() >= p:
        return img, boxes

    img = img.transpose(Image.FLIP_LEFT_RIGHT)
    if boxes is None or len(boxes) == 0:
        return img, boxes

    W, _ = img.size
    boxes = boxes.copy()
    boxes[:, 0] = W - boxes[:, 0]   # cx
    boxes[:, 4] = -boxes[:, 4]      # theta

    # Normalize to [-pi/2, pi/2)
    boxes[:, 4] = np.vectorize(_normalize_angle_rad)(boxes[:, 4])
    return img, boxes


def bflip_det(img, boxes, p=0.5):
    """Vertical flip: y' = H - y, θ' = -θ"""
    if random.random() >= p:
        return img, boxes

    img = img.transpose(Image.FLIP_TOP_BOTTOM)
    if boxes is None or len(boxes) == 0:
        return img, boxes

    _, H = img.size
    boxes = boxes.copy()
    boxes[:, 1] = H - boxes[:, 1]   # cy
    boxes[:, 4] = -boxes[:, 4]      # theta

    boxes[:, 4] = np.vectorize(_normalize_angle_rad)(boxes[:, 4])
    return img, boxes


def rotate_det(img, boxes, p=0.5, angle_range=(-30, 30)):
    """Optional: add rotation augmentation"""
    if random.random() >= p:
        return img, boxes

    angle_deg = random.uniform(*angle_range)
    angle_rad = math.radians(angle_deg)

    # Rotate image
    img = img.rotate(angle_deg, expand=False, fillcolor=0)

    if boxes is None or len(boxes) == 0:
        return img, boxes

    W, H = img.size
    cx_img, cy_img = W / 2, H / 2

    boxes = boxes.copy()
    cx, cy = boxes[:, 0], boxes[:, 1]
    theta = boxes[:, 4]

    # Rotate center around image center
    dx = cx - cx_img
    dy = cy - cy_img
    cos_a = math.cos(angle_rad)
    sin_a = math.sin(angle_rad)
    new_dx = dx * cos_a - dy * sin_a
    new_dy = dx * sin_a + dy * cos_a
    boxes[:, 0] = new_dx + cx_img
    boxes[:, 1] = new_dy + cy_img

    # Add rotation to box angle
    boxes[:, 4] = theta + angle_rad
    boxes[:, 4] = np.vectorize(_normalize_angle_rad)(boxes[:, 4])

    return img, boxes




def resize_(img, boxes, mask, ratio_range):
    """
    Resize image, mask and boxes by scaling the long side randomly.
    boxes: (N, 5) [cx, cy, w, h, theta_rad]
    """
    w, h = img.size
    long_side = max(h, w)

    # Random long side within range
    new_long = random.randint(
        int(long_side * ratio_range[0]),
        int(long_side * ratio_range[1])
    )

    # Keep aspect ratio
    if h > w:
        new_h = new_long
        new_w = int(w * new_long / h + 0.5)
    else:
        new_w = new_long
        new_h = int(h * new_long / w + 0.5)

    scale_x = new_w / w
    scale_y = new_h / h

    img = img.resize((new_w, new_h), Image.BILINEAR)
    mask = mask.resize((new_w, new_h), Image.NEAREST)  # Use NEAREST for masks to avoid interpolation issues

    if boxes is not None and len(boxes) > 0:
        boxes = boxes.copy()
        boxes[:, 0] *= scale_x  # cx
        boxes[:, 1] *= scale_y  # cy
        boxes[:, 2] *= scale_x  # w
        boxes[:, 3] *= scale_y  # h
        # theta unchanged

    return img, boxes, mask


def crop_(img, boxes, mask, size, ignore_value):
    """
    Random crop to (size, size). Pad if needed.
    Keep boxes whose **at least one corner is inside** the crop.
    """
    w, h = img.size
    padw = max(0, size - w)
    padh = max(0, size - h)

    if padw > 0 or padh > 0:
        # Pad image with 0 (black), mask with ignore_value
        img = ImageOps.expand(img, border=(0, 0, padw, padh), fill=0)
        mask = ImageOps.expand(mask, border=(0, 0, padw, padh), fill=ignore_value)
        w, h = img.size

    x = random.randint(0, w - size)
    y = random.randint(0, h - size)
    img = img.crop((x, y, x + size, y + size))
    mask = mask.crop((x, y, x + size, y + size))

    if boxes is None or len(boxes) == 0:
        return img, boxes, mask

    boxes = boxes.copy()
    boxes[:, 0] -= x  # cx
    boxes[:, 1] -= y  # cy

    # --- Check if any corner of rotated box is inside [0, size) ---
    cx, cy, w_box, h_box, theta = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3], boxes[:, 4]
    cos_a = np.cos(theta)
    sin_a = np.sin(theta)

    dx = w_box / 2
    dy = h_box / 2

    # Four corners: top-right, top-left, bottom-left, bottom-right (order doesn't matter for "any")
    corners_x = np.stack([
        cx + dx * cos_a + dy * sin_a,
        cx - dx * cos_a + dy * sin_a,
        cx - dx * cos_a - dy * sin_a,
        cx + dx * cos_a - dy * sin_a,
    ], axis=1)  # (N, 4)

    corners_y = np.stack([
        cy - dx * sin_a + dy * cos_a,
        cy + dx * sin_a + dy * cos_a,
        cy + dx * sin_a - dy * cos_a,
        cy - dx * sin_a - dy * cos_a,
    ], axis=1)

    # A box is kept if ANY of its corners falls inside the crop region [0, size)
    in_crop = (
        (corners_x >= 0) & (corners_x < size) &
        (corners_y >= 0) & (corners_y < size)
    ).any(axis=1)

    boxes = boxes[in_crop]
    return img, boxes, mask


def hflip_(img, boxes, mask, p=0.5):
    """Horizontal flip: x' = W - x, θ' = -θ"""
    if random.random() >= p:
        return img, boxes, mask

    img = img.transpose(Image.FLIP_LEFT_RIGHT)
    mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
    if boxes is None or len(boxes) == 0:
        return img, boxes, mask

    W, _ = img.size
    boxes = boxes.copy()
    boxes[:, 0] = W - boxes[:, 0]   # cx
    boxes[:, 4] = -boxes[:, 4]      # theta

    # Normalize to [-pi/2, pi/2)
    boxes[:, 4] = np.vectorize(_normalize_angle_rad)(boxes[:, 4])
    return img, boxes, mask

def bflip_(img, boxes, mask, p=0.5):
    """Vertical flip: y' = H - y, θ' = -θ"""
    if random.random() >= p:
        return img, boxes, mask

    img = img.transpose(Image.FLIP_TOP_BOTTOM)
    mask = mask.transpose(Image.FLIP_TOP_BOTTOM)
    if boxes is None or len(boxes) == 0:
        return img, boxes, mask

    _, H = img.size
    boxes = boxes.copy()
    boxes[:, 1] = H - boxes[:, 1]   # cy
    boxes[:, 4] = -boxes[:, 4]      # theta

    boxes[:, 4] = np.vectorize(_normalize_angle_rad)(boxes[:, 4])
    return img, boxes, mask






def resize_mm(img, boxes, mask, sar, ratio_range):
    """
    Resize image, mask, sar and boxes by scaling the long side randomly.
    boxes: (N, 5) [cx, cy, w, h, theta_rad]
    """
    w, h = img.size
    long_side = max(h, w)

    # Random long side within range
    new_long = random.randint(
        int(long_side * ratio_range[0]),
        int(long_side * ratio_range[1])
    )

    # Keep aspect ratio
    if h > w:
        new_h = new_long
        new_w = int(w * new_long / h + 0.5)
    else:
        new_w = new_long
        new_h = int(h * new_long / w + 0.5)

    scale_x = new_w / w
    scale_y = new_h / h

    img = img.resize((new_w, new_h), Image.BILINEAR)
    mask = mask.resize((new_w, new_h), Image.NEAREST)
    sar = sar.resize((new_w, new_h), Image.BILINEAR)  # 对SAR图像进行同样缩放

    if boxes is not None and len(boxes) > 0:
        boxes = boxes.copy()
        boxes[:, 0] *= scale_x  # cx
        boxes[:, 1] *= scale_y  # cy
        boxes[:, 2] *= scale_x  # w
        boxes[:, 3] *= scale_y  # h
        # theta unchanged

    return img, boxes, mask, sar


def crop_mm(img, boxes, mask, sar, size, ignore_value):
    """
    Random crop to (size, size). Pad if needed.
    Keep boxes whose **at least one corner is inside** the crop.
    """
    w, h = img.size
    padw = max(0, size - w)
    padh = max(0, size - h)

    if padw > 0 or padh > 0:
        # Pad image with 0 (black), mask with ignore_value
        img = ImageOps.expand(img, border=(0, 0, padw, padh), fill=0)
        mask = ImageOps.expand(mask, border=(0, 0, padw, padh), fill=ignore_value)
        sar = ImageOps.expand(sar, border=(0, 0, padw, padh), fill=0)  # 同样对SAR图像进行填充
        w, h = img.size

    x = random.randint(0, w - size)
    y = random.randint(0, h - size)
    img = img.crop((x, y, x + size, y + size))
    mask = mask.crop((x, y, x + size, y + size))
    sar = sar.crop((x, y, x + size, y + size))  # 对SAR图像进行裁剪

    if boxes is None or len(boxes) == 0:
        return img, boxes, mask, sar

    boxes = boxes.copy()
    boxes[:, 0] -= x  # cx
    boxes[:, 1] -= y  # cy

    # --- Check if any corner of rotated box is inside [0, size) ---
    cx, cy, w_box, h_box, theta = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3], boxes[:, 4]
    cos_a = np.cos(theta)
    sin_a = np.sin(theta)

    dx = w_box / 2
    dy = h_box / 2

    corners_x = np.stack([
        cx + dx * cos_a + dy * sin_a,
        cx - dx * cos_a + dy * sin_a,
        cx - dx * cos_a - dy * sin_a,
        cx + dx * cos_a - dy * sin_a,
    ], axis=1)  # (N, 4)

    corners_y = np.stack([
        cy - dx * sin_a + dy * cos_a,
        cy + dx * sin_a + dy * cos_a,
        cy + dx * sin_a - dy * cos_a,
        cy - dx * sin_a - dy * cos_a,
    ], axis=1)

    in_crop = (
        (corners_x >= 0) & (corners_x < size) &
        (corners_y >= 0) & (corners_y < size)
    ).any(axis=1)

    boxes = boxes[in_crop]
    return img, boxes, mask, sar


def hflip_mm(img, boxes, mask, sar, p=0.5):
    """Horizontal flip: x' = W - x, θ' = -θ"""
    if random.random() >= p:
        return img, boxes, mask, sar

    img = img.transpose(Image.FLIP_LEFT_RIGHT)
    mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
    sar = sar.transpose(Image.FLIP_LEFT_RIGHT)  # 对SAR图像进行水平翻转
    if boxes is None or len(boxes) == 0:
        return img, boxes, mask, sar

    W, _ = img.size
    boxes = boxes.copy()
    boxes[:, 0] = W - boxes[:, 0]   # cx
    boxes[:, 4] = -boxes[:, 4]      # theta

    boxes[:, 4] = np.vectorize(_normalize_angle_rad)(boxes[:, 4])
    return img, boxes, mask, sar


def bflip_mm(img, boxes, mask, sar, p=0.5):
    """Vertical flip: y' = H - y, θ' = -θ"""
    if random.random() >= p:
        return img, boxes, mask, sar

    img = img.transpose(Image.FLIP_TOP_BOTTOM)
    mask = mask.transpose(Image.FLIP_TOP_BOTTOM)
    sar = sar.transpose(Image.FLIP_TOP_BOTTOM)  # 对SAR图像进行垂直翻转
    if boxes is None or len(boxes) == 0:
        return img, boxes, mask, sar

    _, H = img.size
    boxes = boxes.copy()
    boxes[:, 1] = H - boxes[:, 1]   # cy
    boxes[:, 4] = -boxes[:, 4]      # theta

    boxes[:, 4] = np.vectorize(_normalize_angle_rad)(boxes[:, 4])
    return img, boxes, mask, sar