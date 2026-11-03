"""
用 v2 数据集划分生成 ablation_data_v2.npz
从原始 .hdr/.bin 文件读取，只取 VIS 相机的牛油果数据
"""
import os, json, sys
import numpy as np
import spectral
import torch
import torch.nn.functional as F

base_dir = r'D:\HSI_Fruit_Maturity_Detection\datasets'
local_data_dir = r'e:\DeepHS-Fruit\deephs_fruit\data'
annotation_dir = r'e:\DeepHS-Fruit\deephs_fruit\data\annotations'
out_dir = r'e:\DeepHS-Fruit\HSI_Fruit_Maturity_Detection\HSI_Fruit_Maturity_Detection\ablation_study'

def extract_features(json_path, label_enc=None):
    """从标注文件提取特征，只取 avocado + VIS"""
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    record_to_ann = {ann['record_id']: ann for ann in data.get('annotations', [])}
    X_vis, y_vis, fruit_ids = [], [], []
    total, no_ann, not_avo, not_vis, no_file = 0, 0, 0, 0, 0

    for record in data['records']:
        total += 1
        record_id = record['id']
        if record_id not in record_to_ann:
            no_ann += 1
            continue
        if record.get('fruit', '').lower() != 'avocado':
            not_avo += 1
            continue
        if record.get('camera_type', '') != 'VIS':
            not_vis += 1
            continue

        label = record_to_ann[record_id]['ripeness_state']

        hdr_rel = record['files']['header_file']
        hdr_path = os.path.join(local_data_dir, hdr_rel)

        if not os.path.exists(hdr_path):
            no_file += 1
            continue

        bin_path = hdr_path.replace('.hdr', '.bin')

        try:
            img = spectral.envi.open(hdr_path, image=bin_path).load()
            img = np.nan_to_num(img, nan=0.0, posinf=0.0, neginf=0.0)
            img = np.clip(img, -1e10, 1e10)
            if img.ndim < 3:
                continue

            t = torch.tensor(img, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
            t = F.interpolate(t, size=(62, 62), mode='bilinear', align_corners=False).squeeze(0).numpy()
            resized = np.zeros((t.shape[0], 64, 64), dtype=np.float32)
            resized[:, 1:-1, 1:-1] = t

            fruit_id = str(record_id)
            X_vis.append(resized)
            y_vis.append(label)
            fruit_ids.append(fruit_id)
        except Exception as e:
            print(f'  [WARN] failed to load {hdr_rel}: {e}')
            continue

    print(f'  total={total}, no_ann={no_ann}, not_avo={not_avo}, not_vis={not_vis}, no_file={no_file}, kept={len(X_vis)}')
    return np.array(X_vis), np.array(y_vis), np.array(fruit_ids)


# 读取 v2 划分
print("读取训练集...")
X_train, y_train, _ = extract_features(os.path.join(annotation_dir, 'train_only_labeled_v2.json'))

print("\n读取验证集...")
X_val, y_val, val_fruit_ids = extract_features(os.path.join(annotation_dir, 'val_v2.json'))

print("\n读取测试集...")
X_test, y_test, test_fruit_ids = extract_features(os.path.join(annotation_dir, 'test_v2.json'))

print(f"\n{'='*50}")
print(f"Train: {X_train.shape}  labels: {dict(zip(*np.unique(y_train, return_counts=True)))}")
print(f"Val:   {X_val.shape}  labels: {dict(zip(*np.unique(y_val, return_counts=True)))}")
print(f"Test:  {X_test.shape}  labels: {dict(zip(*np.unique(y_test, return_counts=True)))}")
print(f"{'='*50}")

# 计算训练集的归一化参数
train_mean = np.mean(X_train, axis=(0, 2, 3), keepdims=True)
train_std = np.std(X_train, axis=(0, 2, 3), keepdims=True)

# 保存
out_path = os.path.join(out_dir, 'ablation_data_v2.npz')
np.savez_compressed(
    out_path,
    X_train_vis=X_train.astype(np.float32),
    y_train_vis=y_train,
    X_val_vis=X_val.astype(np.float32),
    y_val_vis=y_val,
    X_test_vis=X_test.astype(np.float32),
    y_test_vis=y_test,
    test_fruit_ids=test_fruit_ids,
    train_mean=train_mean.astype(np.float32),
    train_std=train_std.astype(np.float32),
)
print(f"\n已保存到: {out_path}")
print(f"文件大小: {os.path.getsize(out_path)/1024/1024:.1f} MB")
