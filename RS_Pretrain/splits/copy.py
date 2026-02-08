import os, shutil
from tqdm import tqdm

root = "/data1/users/zhengzhiyu/ssl_workplace/dataset/ssl_pretrain"
dst_root = os.path.join(root, "unlabeled/RS4P-1M/masks")
os.makedirs(dst_root, exist_ok=True)

list_path = "/data1/users/zhengzhiyu/ssl_workplace/S5_fulll/S4_Pretrain/splits/pretrain/unlabeled/RS4P-1M-final-1M-mask.txt"

with open(list_path) as f:
    rel_paths = [x.strip() for x in f if x.strip()]

for rel_path in tqdm(rel_paths):
    src = os.path.join(root, rel_path)
    dst = os.path.join(dst_root, os.path.basename(rel_path))
    if not os.path.exists(src):
        print(f"Missing: {src}")
        continue
    try:
        shutil.copy2(src, dst)
    except Exception as e:
        print(f"Copy failed for {src}: {e}")
