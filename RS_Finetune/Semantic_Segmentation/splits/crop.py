# -*- coding: utf-8 -*-

import os
import numpy as np
from PIL import Image
import math

def batch_crop_images(input_dir, clip_size, stride_size, output_dir):
    image_paths = [
        os.path.join(input_dir, f)
        for f in os.listdir(input_dir)
        if f.lower().endswith(('png', 'jpg', 'jpeg', 'tif', 'tiff'))
    ]

    total_crops = 0

    for image_path in image_paths:
        image_name = os.path.basename(image_path)
        print(f"\nProcessing image: {image_name}")

        image = Image.open(image_path)
        image_np = np.array(image)

        if image_np.ndim == 2:
            h, w = image_np.shape
        else:
            h, w, _ = image_np.shape

        num_rows = math.ceil((h - clip_size) / stride_size) + 1
        num_cols = math.ceil((w - clip_size) / stride_size) + 1

        for row in range(num_rows):
            for col in range(num_cols):
                start_y = row * stride_size
                start_x = col * stride_size
                end_y = min(start_y + clip_size, h)
                end_x = min(start_x + clip_size, w)

                # 回退处理，确保裁剪出的图像是 clip_size×clip_size 大小
                if end_x - start_x < clip_size:
                    start_x = max(end_x - clip_size, 0)
                    end_x = start_x + clip_size

                if end_y - start_y < clip_size:
                    start_y = max(end_y - clip_size, 0)
                    end_y = start_y + clip_size

                cropped_image = image_np[start_y:end_y, start_x:end_x]

                os.makedirs(output_dir, exist_ok=True)

                crop_idx = row * num_cols + col
                filename = f"{os.path.splitext(image_name)[0]}_crop_{crop_idx}.png"
                save_path = os.path.join(output_dir, filename)

                Image.fromarray(cropped_image).save(save_path)
                print(f"Saved: {filename} | Coordinates: ({start_x}, {start_y}) -> ({end_x}, {end_y})")

                total_crops += 1

    print(f"Cropping completed. Total cropped patches: {total_crops}")

# 示例调用
input_dir = '/data1/users/zhengzhiyu/ssl_workplace/dataset/ssl_pretrain/VDD/test/src'
clip_size = 512
stride_size = 512
output_dir = '/data1/users/zhengzhiyu/ssl_workplace/dataset/ssl_pretrain/VDD/test/images'
batch_crop_images(input_dir, clip_size, stride_size, output_dir)
