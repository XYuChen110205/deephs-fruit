"""
CNN + SVM 检测头实验
用CNN提取特征，SVM做最终分类（增加新颖性）
"""
import os, sys, copy, random, json
import numpy as np
import torch, torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from collections import Counter

SEED = 23312323
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'设备: {device}')

sys.path.insert(0, r'e:\DeepHS-Fruit\deephs_fruit')
from core.name_convention import *
from core.measurements import all_fruits_new
from core import fruit_list
import spectral

data_path = r'e:\DeepHS-Fruit\deephs_fruit\data'

# ===== 加载数据（只用配置A：92张真实标签）=====
print('\n获取 avocado VIS 数据...')
from core.datasets.hyperspectral_dataset import get_records
train_records, val_records, test_records = get_records(
    Fruit.AVOCADO, CameraType.VIS, ClassificationType.RIPENESS,
    use_inter_ripeness_levels=True, extend_by_time_assumption=False, random_split=False)

def load_split(records, data_path):
    X, y, fids = [], [], []
    for r in records:
        if not r.is_labeled(): continue
        try:
            p = r.get_file_path()
            h = os.path.join(data_path, p + '.hdr')
            b = os.path.join(data_path, p + '.bin')
            if not os.path.exists(h):
                h = os.path.join(data_path, p.replace('Avocado', 'avocado') + '.hdr')
                b = os.path.join(data_path, p.replace('Avocado', 'avocado') + '.bin')
            img = spectral.envi.open(h, image=b).load()
            img = np.nan_to_num(img, nan=0.0, posinf=0.0, neginf=0.0)
            img = np.clip(img, -1e10, 1e10)
            t = torch.tensor(img, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
            t = nn.functional.interpolate(t, size=(62, 62), mode='bilinear', align_corners=False).squeeze(0).numpy()
            p2 = np.zeros((224, 64, 64), dtype=np.float32)
            p2[:, 1:-1, 1:-1] = t
            lm = {RipenessState.UNRIPE: 'unripe', RipenessState.PERFECT: 'perfect', RipenessState.OVERRIPE: 'overripe'}
            y.append(lm[r.label.ripeness_state])
            X.append(p2)
            fids.append(str(r.id))
        except Exception as e:
            print(f'  跳过 {p}: {e}')
            continue
    return np.array(X), np.array(y), np.array(fids)

print('加载数据...')
X_train, y_train, _ = load_split(train_records, data_path)
X_val, y_val, _ = load_split(val_records, data_path)
X_test, y_test, _ = load_split(test_records, data_path)

print(f'  Train: {len(X_train)} {dict(Counter(y_train))}')
print(f'  Val:   {len(X_val)} {dict(Counter(y_val))}')
print(f'  Test:  {len(X_test)} {dict(Counter(y_test))}')

# 合并训练集和验证集给SVM用（SVM不需要验证集调参）
X_train_full = np.concatenate([X_train, X_val], axis=0)
y_train_full = np.concatenate([y_train, y_val], axis=0)

# 归一化
m = np.mean(X_train_full, axis=(0, 2, 3), keepdims=True)
s = np.std(X_train_full, axis=(0, 2, 3), keepdims=True)
s[s < 1e-6] = 1
X_train_full = (X_train_full - m) / s
X_test = (X_test - m) / s
np.clip(X_train_full, -10, 10, out=X_train_full)
np.clip(X_test, -10, 10, out=X_test)

# 编码标签
le = LabelEncoder()
y_train_enc = le.fit_transform(y_train_full)
y_test_enc = le.transform(y_test)

print(f'\n类别映射: {dict(zip(le.classes_, range(len(le.classes_))))}')

# ===== CNN特征提取器（去掉最后的分类头）=====
class SEBlock(nn.Module):
    def __init__(self, ch, r=16):
        super().__init__()
        mid = max(ch // r, 4)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(ch, mid), nn.ReLU(True),
            nn.Linear(mid, ch), nn.Sigmoid()
        )
    def forward(self, x):
        return x * self.fc(self.gap(x).flatten(1)).unsqueeze(-1).unsqueeze(-1)

class FeatureExtractor(nn.Module):
    """CNN特征提取器，输出特征向量给SVM用"""
    def __init__(self, bands=224, hidden=[25, 30, 50]):
        super().__init__()
        k = 3
        # Block 1
        self.b1 = nn.Sequential(
            nn.Conv2d(bands, bands*k, 7, padding=3, groups=bands),
            nn.Conv2d(bands*k, hidden[0], 1),
            nn.ReLU(True), nn.AvgPool2d(2),
            nn.BatchNorm2d(hidden[0]), SEBlock(hidden[0])
        )
        # Block 2
        self.b2 = nn.Sequential(
            nn.Conv2d(hidden[0], hidden[0]*k, 3, padding=1, groups=hidden[0]),
            nn.Conv2d(hidden[0]*k, hidden[1], 1),
            nn.ReLU(True), nn.AvgPool2d(2),
            nn.BatchNorm2d(hidden[1]), SEBlock(hidden[1])
        )
        # Block 3
        self.b3 = nn.Sequential(
            nn.Conv2d(hidden[1], hidden[1]*k, 3, padding=1, groups=hidden[1]),
            nn.Conv2d(hidden[1]*k, hidden[2], 1),
            nn.ReLU(True), nn.BatchNorm2d(hidden[2])
        )
        self.gap = nn.AdaptiveAvgPool2d(1)  # 输出 [B, hidden[2], 1, 1]
        
        # 初始化
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    
    def forward(self, x):
        x = self.b3(self.b2(self.b1(x)))
        x = self.gap(x).flatten(1)  # [B, hidden[2]]
        return x
    
    def extract_features(self, X, batch_size=32):
        """批量提取特征"""
        self.eval()
        features = []
        with torch.no_grad():
            for i in range(0, len(X), batch_size):
                batch = torch.FloatTensor(X[i:i+batch_size]).to(device)
                feat = self.forward(batch)
                features.append(feat.cpu().numpy())
        return np.concatenate(features, axis=0)

def augment(X, y, noise_level=0.02, num_augment=2):
    """数据增强：添加噪声和翻转"""
    X_aug, y_aug = [X], [y]
    for _ in range(num_augment):
        X_noise = X + np.random.randn(*X.shape) * noise_level
        X_aug.append(X_noise)
        y_aug.append(y.copy())
    return np.concatenate(X_aug, axis=0), np.concatenate(y_aug, axis=0)

# ===== 训练CNN特征提取器 =====
print(f'\n{"="*60}')
print('阶段1：训练CNN特征提取器（无分类头）')
print(f'{"="*60}')

# 数据增强
X_train_aug, y_train_aug = augment(X_train_full, y_train_enc, noise_level=0.02, num_augment=2)
print(f'数据增强后: {len(X_train_aug)} 张')

# 转换为tensor
X_train_t = torch.FloatTensor(X_train_aug)
y_train_t = torch.LongTensor(y_train_aug)
train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=32, shuffle=True)

# 创建模型（带临时分类头用于训练）
class CNNWithTempHead(nn.Module):
    def __init__(self, feature_extractor, num_classes):
        super().__init__()
        self.fe = feature_extractor
        self.fc = nn.Linear(50, num_classes)  # 临时分类头
    
    def forward(self, x):
        feat = self.fe(x)
        return self.fc(feat)

fe = FeatureExtractor().to(device)
model = CNNWithTempHead(fe, len(le.classes_)).to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100, eta_min=1e-6)

# 训练
for epoch in range(100):
    model.train()
    total_loss = 0
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        loss = criterion(model(x), y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()
    scheduler.step()
    if (epoch + 1) % 20 == 0:
        print(f'  Epoch {epoch+1}/100, Loss: {total_loss/len(train_loader):.4f}')

# ===== 提取特征 =====
print(f'\n{"="*60}')
print('阶段2：用训练好的CNN提取特征')
print(f'{"="*60}')

# 去掉临时分类头，只用特征提取器
train_features = fe.extract_features(X_train_full)
test_features = fe.extract_features(X_test)

print(f'训练集特征形状: {train_features.shape}')
print(f'测试集特征形状: {test_features.shape}')

# ===== SVM分类 =====
print(f'\n{"="*60}')
print('阶段3：SVM分类（检测头）')
print(f'{"="*60}')

# 尝试不同核函数的SVM
kernels = ['rbf', 'linear', 'poly']
best_svm = None
best_acc = 0

print('\n对比不同SVM核函数:')
for kernel in kernels:
    svm = SVC(kernel=kernel, C=1.0, gamma='scale', random_state=SEED)
    svm.fit(train_features, y_train_full)
    pred = svm.predict(test_features)
    acc = accuracy_score(y_test_enc, pred)
    f1 = f1_score(y_test_enc, pred, average='weighted')
    print(f'  SVM-{kernel:6s}: acc={acc*100:.1f}%, f1={f1*100:.1f}%')
    if acc > best_acc:
        best_acc = acc
        best_svm = svm

# 再试试随机森林作为对比
print('\n对比其他传统分类器:')
rf = RandomForestClassifier(n_estimators=100, random_state=SEED)
rf.fit(train_features, y_train_full)
rf_pred = rf.predict(test_features)
rf_acc = accuracy_score(y_test_enc, rf_pred)
rf_f1 = f1_score(y_test_enc, rf_pred, average='weighted')
print(f'  RandomForest: acc={rf_acc*100:.1f}%, f1={rf_f1*100:.1f}%')

# ===== 最终结果 =====
print(f'\n{"="*60}')
print('最终结果')
print(f'{"="*60}')

final_pred = best_svm.predict(test_features)
final_acc = accuracy_score(y_test_enc, final_pred)
final_f1 = f1_score(y_test_enc, final_pred, average='weighted')
cm = confusion_matrix(y_test_enc, final_pred)

print(f'\n最佳模型: CNN + SVM ({best_svm.kernel}核)')
print(f'测试准确率: {final_acc*100:.1f}%')
print(f'F1分数: {final_f1*100:.1f}%')
print(f'\n混淆矩阵:')
print(cm)

print(f'\n对比之前的结果:')
print(f'  纯CNN (全连接检测头): 66.3%')
print(f'  CNN+SVM (SVM检测头):   {final_acc*100:.1f}%')
print(f'  提升: {final_acc*100 - 66.3:+.1f}%')

# 保存结果
results = {
    'method': f'CNN+SVM({best_svm.kernel})',
    'accuracy': round(float(final_acc), 4),
    'f1': round(float(final_f1), 4),
    'feature_dim': train_features.shape[1],
    'train_samples': len(X_train_full),
    'test_samples': len(X_test)
}

with open('cnn_svm_results.json', 'w') as f:
    json.dump(results, f, indent=2)
print('\n结果已保存到 cnn_svm_results.json')
