
import os
import json
import numpy as np
import spectral
import torch
import torch.nn.functional as F
from tqdm import tqdm

def extract_features_2d_raw(json_path, base_dir, fruit_name='avocado'):
    with open(json_path, 'r') as f:
        data = json.load(f)
    record_to_ann = {ann['record_id']: ann for ann in data.get('annotations', [])}
    X_vis, y_vis, fruit_ids = [], [], []
    for record in tqdm(data['records'], desc=f"Processing {os.path.basename(json_path)}"):
        record_id = record['id']
        if record_id not in record_to_ann or record['fruit'].lower() != fruit_name:
            continue
        label = record_to_ann[record_id]['ripeness_state']
        hdr_path = os.path.join(base_dir, record['files']['header_file'])
        bin_path = os.path.join(base_dir, record['files']['data_file'])
        if not os.path.exists(hdr_path) or not os.path.exists(bin_path):
            continue
        try:
            img = spectral.envi.open(hdr_path, image=bin_path).load()
            img = np.nan_to_num(img)
            if img.ndim &lt; 3:
                continue
            t = torch.tensor(img, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
            t = F.interpolate(t, size=(62, 62), mode='bilinear', align_corners=False).squeeze(0).numpy()
            resized = np.zeros((t.shape[0], 64, 64), dtype=np.float32)
            resized[:, 1:-1, 1:-1] = t
            fruit_id = os.path.basename(hdr_path).split('_')[-2]
            X_vis.append(resized)
            y_vis.append(label)
            fruit_ids.append(fruit_id)
        except Exception as e:
            continue
    return np.array(X_vis), np.array(y_vis), np.array(fruit_ids)

base_dir = r'D:\HSI_Fruit_Maturity_Detection\datasets'

X_train_raw, y_train_vis, _ = extract_features_2d_raw(
    os.path.join(base_dir, 'annotations', 'train_only_labeled.json'), base_dir)
X_val_raw, y_val_vis, _ = extract_features_2d_raw(
    os.path.join(base_dir, 'annotations', 'val.json'), base_dir)
X_test_raw, y_test_vis, test_fruit_ids = extract_features_2d_raw(
    os.path.join(base_dir, 'annotations', 'test.json'), base_dir)

X_train_raw = np.nan_to_num(X_train_raw, nan=0.0, posinf=0.0, neginf=0.0)
X_val_raw   = np.nan_to_num(X_val_raw,   nan=0.0, posinf=0.0, neginf=0.0)
X_test_raw  = np.nan_to_num(X_test_raw,  nan=0.0, posinf=0.0, neginf=0.0)

p01 = np.percentile(X_train_raw, 1)
p99 = np.percentile(X_train_raw, 99)
X_train_raw = np.clip(X_train_raw, p01, p99)
X_val_raw   = np.clip(X_val_raw,   p01, p99)
X_test_raw  = np.clip(X_test_raw,  p01, p99)

train_mean = X_train_raw.mean(axis=(0, 2, 3), keepdims=True)[0]
train_std  = X_train_raw.std( axis=(0, 2, 3), keepdims=True)[0] + 1e-8
X_train_vis = (X_train_raw - train_mean) / train_std
X_val_vis   = (X_val_raw   - train_mean) / train_std
X_test_vis  = (X_test_raw  - train_mean) / train_std

X_train_vis = np.clip(np.nan_to_num(X_train_vis), -10, 10)
X_val_vis   = np.clip(np.nan_to_num(X_val_vis),   -10, 10)
X_test_vis  = np.clip(np.nan_to_num(X_test_vis),  -10, 10)

cache_path = 'ablation_data.npz'
np.savez(cache_path,
         X_train=X_train_vis, y_train=y_train_vis,
         X_val=X_val_vis, y_val=y_val_vis,
         X_test=X_test_vis, y_test=y_test_vis,
         test_fruit_ids=test_fruit_ids,
         train_mean=train_mean, train_std=train_std)
print(f'Data saved to {cache_path}')
print(f'Train: {X_train_vis.shape}, Val: {X_val_vis.shape}, Test: {X_test_vis.shape}')

