# import torch

# # # ckpt_path = "/data1/users/zhengzhiyu/ssl_workplace/semi_sep/pretrained/semi_sep/vit_l/rf_vit_l_upernet_semi_sep_iter_75000.pth"
# ckpt_path = "/data1/users/zhengzhiyu/ssl_workplace/S5_fulll/S4_Pretrain/pretrained/vit_l_s4p.pth"
# save_path = "/data1/users/zhengzhiyu/ssl_workplace/S5_fulll/S4_Pretrain/pretrained/vit_b_s4p.pth"

# checkpoint = torch.load(ckpt_path, map_location="cpu")

# # 只保留 "model"
# new_ckpt = {"model": checkpoint["model"]}

# torch.save(new_ckpt, save_path)
# print(f"Cleaned checkpoint saved at {save_path}")



# import os
# from pathlib import Path

# src_dir = Path("/data1/users/zhengzhiyu/ssl_workplace/semi_sep2/splits/pretrain/unlabeled/cluster_vit_h_1m")
# dst_file = Path("/data1/users/zhengzhiyu/ssl_workplace/semi_sep2/splits/pretrain/unlabeled/RS4P-1M.txt")

# with open(dst_file, "w") as fout:
#     for txt_path in sorted(src_dir.glob("*.txt")):  # 按字母顺序合并，可选
#         with open(txt_path, "r") as fin:
#             for line in fin:
#                 fout.write(line)
