import os
from PIL import Image

txt_path = '/data1/users/zhengzhiyu/ssl_workplace/S5_fulll/S4_Pretrain/splits/pretrain_mt/unlabeled/SARLO.txt'
data_root = '/data2/users/yangcong2/sslworkplace/dataset/'

broken_files = []

def is_image_valid(path):
    """尝试打开并加载图像，返回 True/False"""
    try:
        with Image.open(path) as img:
            img.load()  # 触发实际解码，检测损坏
        return True
    except Exception as e:
        return False

def main():
    with open(txt_path, 'r') as f:
        lines = [line.strip() for line in f if line.strip()]

    total_pairs = len(lines)
    total_images = total_pairs * 2
    print(f"🔍 Checking {total_pairs} pairs ({total_images} images)...")

    for idx, line in enumerate(lines, 1):
        parts = line.split()
        if len(parts) != 2:
            print(f"⚠️ Skipping invalid line #{idx}: {line}")
            continue

        optic_rel, sar_rel = parts
        optic_path = os.path.join(data_root, optic_rel)
        sar_path = os.path.join(data_root, sar_rel)

        # Check optic
        if not is_image_valid(optic_path):
            broken_files.append(optic_path)

        # Check SAR
        if not is_image_valid(sar_path):
            broken_files.append(sar_path)

        if idx % 100 == 0:
            print(f"✅ Processed {idx}/{total_pairs} pairs")

    # Final report
    if broken_files:
        print("\n❌ Found broken images:")
        for p in broken_files:
            print(p)
        print(f"\n💥 Total broken images: {len(broken_files)} out of {total_images}")
    else:
        print(f"\n🎉 All {total_images} images are valid! No broken files found.")

if __name__ == '__main__':
    main()