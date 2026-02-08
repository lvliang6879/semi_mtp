import os

instance_base_directory = '/data1/users/zhengzhiyu/ssl_workplace/dataset/ssl_pretrain/'

imgs_directory = os.path.join(instance_base_directory, 'labeled/MOTA/split_ms_dota1_0/val/images/')
det_labels_directory = os.path.join(instance_base_directory, 'labeled/MOTA/split_ms_dota1_0/val/labelTxt/')
seg_masks_directory = os.path.join(instance_base_directory, 'labeled/MOTA/split_ms_dota1_0/val/merged_IRSAMap_masks/')  # 新增

output_file = '/data1/users/zhengzhiyu/ssl_workplace/S5_fulll/S4_Pretrain/splits/pretrain_mt/labeled/labeled_merge_IRSAMap.txt'

# 支持的图像文件扩展名
image_extensions = {'.jpg', '.jpeg', '.png', '.tif'}

def get_image_det_seg_triples(imgs_dir, det_dir, seg_dir):
    triples = []
    
    # 获取所有图像文件（过滤扩展名）
    img_files = sorted([
        f for f in os.listdir(imgs_dir)
        if os.path.splitext(f)[1].lower() in image_extensions
    ])
    
    print('First 5 image files:', img_files[:5])
    
    for img_filename in img_files:
        base_name = os.path.splitext(img_filename)[0]  # e.g., 'P0001'
        
        # 构造对应的检测标签和分割掩码文件名
        det_filename = base_name + '.txt'
        seg_filename = base_name + '.png'  # 或 .tif, 根据你的格式调整
        
        img_path = os.path.join(imgs_dir, img_filename)
        det_path = os.path.join(det_dir, det_filename)
        seg_path = os.path.join(seg_dir, seg_filename)
        
        # 检查三个文件是否存在
        if not os.path.exists(det_path):
            print(f"Warning: detection label missing for {img_filename}")
            continue
        if not os.path.exists(seg_path):
            print(f"Warning: segmentation mask missing for {img_filename}")
            continue
        
        # 转为相对于 instance_base_directory 的路径
        rel_img = os.path.relpath(img_path, instance_base_directory)
        rel_det = os.path.relpath(det_path, instance_base_directory)
        rel_seg = os.path.relpath(seg_path, instance_base_directory)
        
        triples.append(f"{rel_img} {rel_det} {rel_seg}")
    
    return triples

# 获取三元组
triples = get_image_det_seg_triples(imgs_directory, det_labels_directory, seg_masks_directory)

print(f"Total valid triples collected: {len(triples)}")

# 读取现有文件内容（去重用）
existing_lines = set()
if os.path.exists(output_file):
    with open(output_file, 'r') as f:
        existing_lines = set(line.strip() for line in f if line.strip())

print(f"Existing lines in file: {len(existing_lines)}")

# 合并并去重
all_lines = sorted(set(existing_lines).union(triples))
print(f"Total unique lines to write: {len(all_lines)}")

# 创建输出目录
os.makedirs(os.path.dirname(output_file), exist_ok=True)

# 写入文件
with open(output_file, 'w') as f:
    for line in all_lines:
        f.write(line + '\n')

print("Finished writing image-detection-segmentation triples to file.")