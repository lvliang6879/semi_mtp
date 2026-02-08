import os

# 根目录配置
instance_base_directory = '/data2/users/lanjie/proj/SSL_workplace/dataset/'
dataset_root = os.path.join(instance_base_directory, 'OpenEarthMap_with_xBD')
train_list_path = os.path.join(dataset_root, 'test.txt')
output_file = '/data2/users/lanjie/proj/SSL_workplace/S5/splits/OpenEarthMap/test.txt'

# 从 train.txt 中读取图像文件名（例如 aachen_1.tif）
with open(train_list_path, 'r') as f:
    image_names = set(line.strip() for line in f if line.strip())

print(f"✅ train.txt 中图像数量: {len(image_names)}")

pairs = []

# 遍历 OpenEarthMap_with_xBD 下的所有子目录（如 aachen、zanzibar 等）
for subdir_name in os.listdir(dataset_root):
    sub_path = os.path.join(dataset_root, subdir_name)
    if not os.path.isdir(sub_path):
        continue

    images_dir = os.path.join(sub_path, 'images')
    labels_dir = os.path.join(sub_path, 'images')

    if not os.path.isdir(images_dir) or not os.path.isdir(labels_dir):
        continue

    for img_file in os.listdir(images_dir):
        if img_file in image_names:
            abs_img = os.path.join(images_dir, img_file)
            abs_lbl = os.path.join(labels_dir, img_file)

            # 确保 label 文件存在
            if os.path.exists(abs_lbl):
                # 计算相对路径（相对于 dataset_root）  
                prefix = 'OpenEarthMap_with_xBD/'
                rel_img = prefix + os.path.relpath(abs_img, dataset_root)
                rel_lbl = prefix + os.path.relpath(abs_lbl, dataset_root)
                pairs.append(f"{rel_img} {rel_lbl}")
            else:
                print(f"[警告] 找不到 label 文件: {abs_lbl}")

print(f"✅ 成功配对图像-标签: {len(pairs)}")

# 确保输出目录存在
os.makedirs(os.path.dirname(output_file), exist_ok=True)

# 写入结果
with open(output_file, 'w') as f:
    for p in pairs:
        f.write(p + '\n')

print("✅ labeled.txt 写入完成 ✅")
