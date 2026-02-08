import os

instance_base_directory = '/data1/users/zhengzhiyu/mtp_workplace/dataset/oscd_rgb_patch_dataset/test/A/'
# imgs_directory = os.path.join(instance_base_directory, 'A/')
imgs_directory = instance_base_directory
output_file = '/data1/users/zhengzhiyu/ssl_workplace/S5/splits/oscd/test.txt'
def get_image_paths(imgs_directory):
    image_paths = []
    img_files = sorted(os.listdir(imgs_directory))

    for img_filename in img_files:
        rel_img_path = os.path.relpath(os.path.join(imgs_directory, img_filename), instance_base_directory)
        image_paths.append(rel_img_path)

    return image_paths

# 获取图像相对路径
image_paths = get_image_paths(imgs_directory)
print(f"Total images found: {len(image_paths)}")

# 读取现有文件内容（如果文件存在）
existing_paths = set()
if os.path.exists(output_file):
    with open(output_file, 'r') as f:
        existing_paths = set(f.read().splitlines())

print(f"Existing paths in file: {len(existing_paths)}")

# 合并去重
all_paths = sorted(set(existing_paths).union(image_paths))
print(f"Total unique paths to write: {len(all_paths)}")

# 创建目录（如不存在）
os.makedirs(os.path.dirname(output_file), exist_ok=True)

# 写入文件
with open(output_file, 'w') as f:
    for path in all_paths:
        f.write(f"{path}\n")

print("Finished writing image paths to file.")
