import os
import json
import numpy as np
import spectral
import torch
import torch.nn.functional as F
from tqdm import tqdm
from avocado_detection.config import base_dir


def extract_features_2d_raw(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    record_to_ann = {ann['record_id']: ann for ann in data.get('annotations', [])}
    X_vis, y_vis, fruit_ids = [], [], []
    for record in tqdm(data['records'], desc=f"Processing {os.path.basename(json_path)}"):
        record_id = record['id']
        if record_id not in record_to_ann or record['fruit'].lower() != 'avocado': continue
        label = record_to_ann[record_id]['ripeness_state']
        if record['camera_type'] != 'VIS': continue
        hdr_path = os.path.join(base_dir, record['files']['header_file'])
        bin_path = os.path.join(base_dir, record['files']['data_file'])
        if not os.path.exists(hdr_path) or not os.path.exists(bin_path): continue
        try:
            img = spectral.envi.open(hdr_path, image=bin_path).load()
            img = np.nan_to_num(img)
            if img.ndim < 3: continue
            t = torch.tensor(img, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
            t = F.interpolate(t, size=(62, 62), mode='bilinear', align_corners=False).squeeze(0).numpy()
            resized = np.zeros((t.shape[0], 64, 64), dtype=np.float32)
            resized[:, 1:-1, 1:-1] = t
            fruit_id = os.path.basename(hdr_path).split('_')[-2]
            X_vis.append(resized); y_vis.append(label); fruit_ids.append(fruit_id)
        except Exception: continue
    return np.array(X_vis), np.array(y_vis), np.array(fruit_ids)


def diagnose_dataset(y_train_enc, y_val_enc, y_test_enc, le):
    class_names = le.classes_
    print("\n" + "="*65)
    print("数据集诊断")
    print("="*65)
    for split_name, y_enc in [("Train", y_train_enc),
                               ("Val",   y_val_enc),
                               ("Test",  y_test_enc)]:
        counts = np.bincount(y_enc, minlength=len(class_names))
        total  = len(y_enc)
        print(f"\n  {split_name:5s}  总计: {total} 样本")
        for cls_idx, (name, cnt) in enumerate(zip(class_names, counts)):
            bar = '█' * int(cnt / max(counts) * 20)
            print(f"    [{cls_idx}] {name:12s}: {cnt:4d} ({cnt/total*100:5.1f}%)  {bar}")
    print("="*65 + "\n")
    val_counts = np.bincount(y_val_enc, minlength=len(class_names))
    if val_counts.min() == 0:
        print("⚠️  警告：验证集中有类别样本数为 0！")
    if len(y_val_enc) < 30:
        print(f"⚠️  警告：验证集仅 {len(y_val_enc)} 个样本，acc 会大幅跳动，建议增大验证集。")
    train_counts = np.bincount(y_train_enc, minlength=len(class_names))
    imbalance = train_counts.max() / max(train_counts.min(), 1)
    if imbalance > 5:
        print(f"⚠️  警告：训练集类别不平衡比 {imbalance:.1f}x，WeightedSampler 已启用。")
