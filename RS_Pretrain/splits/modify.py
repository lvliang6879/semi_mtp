# 打开原始文件路径txt
with open('/data1/users/zhengzhiyu/ssl_workplace/S5_fulll/S4_Pretrain/splits/pretrain_mt/val_mota_ss.txt', 'r') as file:
    lines = file.readlines()

# 创建一个新的文件来保存修改后的内容
with open('/data1/users/zhengzhiyu/ssl_workplace/S5_fulll/S4_Pretrain/splits/pretrain_mt/val_merge_IRSAMap.txt', 'w') as file:
    for line in lines:
        # 替换路径中的 'supervised/Instance/Isaid/' 为 'labeled/iSAID'
        modified_line = line.replace('new_labels_255', 'merged_IRSAMap_masks' )
        # 写入修改后的行
        file.write(modified_line)

print("路径修改完成，保存到 modified_paths.txt")
