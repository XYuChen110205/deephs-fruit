#!/usr/bin/env python3
"""
多种子全面实验 — 用于竞赛报告数据支撑
使用 ablation_data_v2.npz (83个测试样本,含 test_fruit_ids)
跑 3 个种子,输出 mean ± std,保存完整结果到 JSON

实验内容:
  1. 消融实验 (4种配置 x 3种子 = 12次)
  2. 对比实验 (KNN/SVM/RF/MLP x 3种子 = 12次)
  3. 我们的完整模型 (3种子)
"""
import os
import sys
import copy
import json
import random
import warnings
from collections import defaultdict
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.metrics import classification_report, confusion_matrix

warnings.filterwarnings('ignore')

# ===========================================================
# 配置
# ===========================================================
SEEDS = [1, 2, 42]
DATA_PATH = os.path.join(os.path.dirname(__file__), 'ablation_study', 'ablation_data_v2.npz')

BASE_LR = 1e-3
MIN_LR = 1e-6
WEIGHT_DECAY = 1e-4
BATCH_SIZE = 32
EPOCHS = 150
PATIENCE = 30
LR_WARMUP_EPOCHS = 10
GRAD_CLIP = 1.0
EMA_WARMUP_EPOCHS = 15
EMA_INIT_DECAY = 0.9
EMA_FINAL_DECAY = 0.999
TTA_ITER = 5

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'experiment_results')
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ===========================================================
# 模型定义
# ===========================================================
class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        mid = max(channels // reduction, 4)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, mid), nn.ReLU(True),
            nn.Linear(mid, channels), nn.Sigmoid()
        )
    def forward(self, x):
        w = self.fc(self.gap(x).flatten(1)).unsqueeze(-1).unsqueeze(-1)
        return x * w


class IdentityBlock(nn.Module):
    def forward(self, x): return x


def make_model(bands, num_classes, use_se=True):
    """构建模型,use_se=False 时用 IdentityBlock 替换 SEBlock"""
    hidden = [25, 30, 50]
    k = 3
    attn1 = SEBlock(hidden[0]) if use_se else IdentityBlock()
    attn2 = SEBlock(hidden[1]) if use_se else IdentityBlock()

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.block1 = nn.Sequential(
                nn.Conv2d(bands, bands*k, 7, padding=3, groups=bands),
                nn.Conv2d(bands*k, hidden[0], 1),
                nn.ReLU(True), nn.AvgPool2d(2),
                nn.BatchNorm2d(hidden[0]), attn1,
            )
            self.block2 = nn.Sequential(
                nn.Conv2d(hidden[0], hidden[0]*k, 3, padding=1, groups=hidden[0]),
                nn.Conv2d(hidden[0]*k, hidden[1], 1),
                nn.ReLU(True), nn.AvgPool2d(2),
                nn.BatchNorm2d(hidden[1]), attn2,
            )
            self.block3 = nn.Sequential(
                nn.Conv2d(hidden[1], hidden[1]*k, 3, padding=1, groups=hidden[1]),
                nn.Conv2d(hidden[1]*k, hidden[2], 1),
                nn.ReLU(True), nn.BatchNorm2d(hidden[2]),
            )
            self.gap = nn.AdaptiveAvgPool2d(1)
            self.fc = nn.Sequential(
                nn.Linear(hidden[2], hidden[2]),
                nn.BatchNorm1d(hidden[2]),
                nn.ReLU(True), nn.Dropout(0.3),
                nn.Linear(hidden[2], num_classes)
            )
            self._init_weights()
        def _init_weights(self):
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                    if m.bias is not None: nn.init.zeros_(m.bias)
                elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                    nn.init.ones_(m.weight); nn.init.zeros_(m.bias)
                elif isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, std=0.01)
                    if m.bias is not None: nn.init.zeros_(m.bias)
        def forward(self, x):
            return self.fc(self.gap(self.block3(self.block2(self.block1(x)))).flatten(1))
    return Net()


# ===========================================================
# 工具
# ===========================================================
class EMA:
    def __init__(self, model, init_decay=0.9):
        self.decay = init_decay
        self.shadow = copy.deepcopy(model)
        self.shadow.eval()
        for p in self.shadow.parameters(): p.requires_grad_(False)
    @torch.no_grad()
    def update(self, model):
        for s_p, m_p in zip(self.shadow.parameters(), model.parameters()):
            s_p.data.mul_(self.decay).add_(m_p.data, alpha=1.0 - self.decay)
        for s_b, m_b in zip(self.shadow.buffers(), model.buffers()):
            s_b.copy_(m_b)
    def set_decay(self, d): self.decay = d
    def eval_model(self): return self.shadow


class WarmupCosineScheduler:
    def __init__(self, optimizer, warmup, total, base_lr, min_lr=1e-6):
        self.opt, self.warmup, self.total = optimizer, warmup, total
        self.base_lr, self.min_lr, self.e = base_lr, min_lr, 0
    def step(self):
        e = self.e
        if e < self.warmup:
            lr = self.base_lr * (e + 1) / self.warmup
        else:
            p = (e - self.warmup) / max(self.total - self.warmup, 1)
            lr = self.min_lr + 0.5 * (self.base_lr - self.min_lr) * (1 + np.cos(np.pi * p))
        for pg in self.opt.param_groups: pg['lr'] = lr
        self.e += 1
        return lr


def augment_batch(x):
    out = x.clone()
    for i in range(len(out)):
        if torch.rand(1) > 0.5: out[i] = torch.flip(out[i], [1])
        if torch.rand(1) > 0.5: out[i] = torch.flip(out[i], [2])
        if torch.rand(1) > 0.5: out[i] = torch.rot90(out[i], 1, [1, 2])
        if torch.rand(1) > 0.5: out[i] += torch.randn_like(out[i]) * 0.02
    return out


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    loss, correct, total = 0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        pred = model(x)
        loss += criterion(pred, y).item()
        correct += (pred.argmax(1) == y).sum().item()
        total += y.size(0)
    return loss / len(loader), correct / total


def fruit_level_acc(probs, fruit_ids, y_true):
    d, labels = defaultdict(list), {}
    for i, fid in enumerate(fruit_ids):
        d[fid].append(probs[i]); labels[fid] = y_true[i]
    preds = [np.stack(v).mean(0).argmax() for v in d.values()]
    return accuracy_score(list(labels.values()), preds)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ===========================================================
# 训练函数
# ===========================================================
def train_one_config(X_train, y_tr, X_val, y_v, X_test, y_te, test_fruit_ids,
                     le, seed, use_se=True, use_ema=True, use_tta=True,
                     use_label_smooth=True, use_aug=True, use_warmup=True):
    """训练单次实验,返回完整结果字典"""
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    X_train_t = torch.FloatTensor(X_train); y_train_t = torch.LongTensor(y_tr)
    X_val_t   = torch.FloatTensor(X_val);   y_val_t   = torch.LongTensor(y_v)
    X_test_t  = torch.FloatTensor(X_test);  y_test_t  = torch.LongTensor(y_te)

    # 加权采样
    class_counts = np.bincount(y_tr)
    weights = 1.0 / class_counts
    sampler = torch.utils.data.WeightedRandomSampler(weights[y_tr], len(y_tr))

    train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=BATCH_SIZE, sampler=sampler)
    val_loader   = DataLoader(TensorDataset(X_val_t, y_val_t), batch_size=BATCH_SIZE, shuffle=False)
    test_loader  = DataLoader(TensorDataset(X_test_t, y_test_t), batch_size=BATCH_SIZE, shuffle=False)

    num_classes = len(le.classes_)
    bands = X_train.shape[1]
    model = make_model(bands, num_classes, use_se=use_se).to(device)

    train_crit = nn.CrossEntropyLoss(label_smoothing=0.1 if use_label_smooth else 0.0)
    val_crit   = nn.CrossEntropyLoss()

    optimizer = optim.AdamW(model.parameters(), lr=BASE_LR, weight_decay=WEIGHT_DECAY)
    if use_warmup:
        scheduler = WarmupCosineScheduler(optimizer, LR_WARMUP_EPOCHS, EPOCHS, BASE_LR, MIN_LR)
    else:
        scheduler = None

    ema = EMA(model, init_decay=EMA_INIT_DECAY) if use_ema else None

    best_val_loss = float('inf')
    best_model_state = None
    best_ema_state = None
    patience_counter = 0
    final_epoch = 0

    for epoch in range(EPOCHS):
        model.train()
        if scheduler: scheduler.step()

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            if use_aug: x = augment_batch(x)
            optimizer.zero_grad()
            loss = train_crit(model(x), y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRAD_CLIP)
            optimizer.step()
            if ema: ema.update(model)

        # EMA decay 调度
        if ema:
            if epoch < EMA_WARMUP_EPOCHS:
                ema.set_decay(EMA_INIT_DECAY)
            else:
                ramp = min((epoch - EMA_WARMUP_EPOCHS) / max(EPOCHS - EMA_WARMUP_EPOCHS, 1), 1.0)
                ema.set_decay(EMA_INIT_DECAY + ramp * (EMA_FINAL_DECAY - EMA_INIT_DECAY))

        # 验证
        if ema and epoch >= EMA_WARMUP_EPOCHS:
            val_loss, val_acc = evaluate(ema.eval_model(), val_loader, val_crit, device)
        else:
            val_loss, val_acc = evaluate(model, val_loader, val_crit, device)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = copy.deepcopy(model.state_dict())
            if ema: best_ema_state = copy.deepcopy(ema.shadow.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1

        final_epoch = epoch + 1
        if patience_counter >= PATIENCE:
            break

    # 推理模型
    if ema and best_ema_state:
        ema.shadow.load_state_dict(best_ema_state)
        inf_model = ema.eval_model()
    else:
        model.load_state_dict(best_model_state)
        inf_model = model
    inf_model.eval()

    # TTA 推理
    tta_iter = TTA_ITER if use_tta else 1
    all_preds = torch.zeros(tta_iter, len(X_test_t), num_classes)
    with torch.no_grad():
        for i in range(tta_iter):
            preds_list = []
            for x, _ in test_loader:
                x = x.to(device)
                if i != 0:
                    x = x.clone()
                    for j in range(len(x)):
                        if torch.rand(1) > 0.5: x[j] = torch.flip(x[j], [1])
                        if torch.rand(1) > 0.5: x[j] = torch.flip(x[j], [2])
                        if torch.rand(1) > 0.5: x[j] = torch.rot90(x[j], 1, [1, 2])
                preds_list.append(F.softmax(inf_model(x), dim=1).cpu())
            all_preds[i] = torch.cat(preds_list, dim=0)

    mean_probs = all_preds.mean(0).numpy()
    final_preds = mean_probs.argmax(1)

    # 指标计算
    sample_acc = accuracy_score(y_te, final_preds)
    fruit_acc  = fruit_level_acc(mean_probs, test_fruit_ids, y_te)
    f1  = f1_score(y_te, final_preds, average='weighted')
    pre = precision_score(y_te, final_preds, average='weighted', zero_division=0)
    rec = recall_score(y_te, final_preds, average='weighted', zero_division=0)
    cm  = confusion_matrix(y_te, final_preds).tolist()
    cr  = classification_report(y_te, final_preds, target_names=le.classes_, output_dict=True)

    params = sum(p.numel() for p in inf_model.parameters())

    return {
        'seed': seed,
        'epochs': final_epoch,
        'sample_acc': float(sample_acc),
        'fruit_acc': float(fruit_acc),
        'f1': float(f1),
        'precision': float(pre),
        'recall': float(rec),
        'params': params,
        'confusion_matrix': cm,
        'per_class': {name: cr[name] for name in le.classes_ if name in cr},
    }


# ===========================================================
# 经典 ML 对比
# ===========================================================
def run_ml_comparison(X_train, y_tr, X_test, y_te, test_fruit_ids, le, seed):
    """KNN / SVM / RF 对比实验"""
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.svm import SVC
    from sklearn.ensemble import RandomForestClassifier

    set_seed(seed)

    # 输入特征: 对每张图取空间均值,得到 224 维光谱向量
    X_train_flat = X_train.mean(axis=(2, 3))  # (N, 224)
    X_test_flat  = X_test.mean(axis=(2, 3))

    results = {}

    methods = {
        'KNN': KNeighborsClassifier(n_neighbors=5, weights='distance'),
        'SVM_RBF': SVC(kernel='rbf', C=10, gamma='scale', probability=True, random_state=seed),
        'RandomForest': RandomForestClassifier(n_estimators=100, max_depth=10, random_state=seed),
    }

    for name, clf in methods.items():
        clf.fit(X_train_flat, y_tr)
        preds = clf.predict(X_test_flat)

        if hasattr(clf, 'predict_proba'):
            probs = clf.predict_proba(X_test_flat)
            f_acc = fruit_level_acc(probs, test_fruit_ids, y_te)
        else:
            f_acc = None

        sample_acc = accuracy_score(y_te, preds)
        f1  = f1_score(y_te, preds, average='weighted')
        pre = precision_score(y_te, preds, average='weighted', zero_division=0)
        rec = recall_score(y_te, preds, average='weighted', zero_division=0)

        results[name] = {
            'seed': seed,
            'sample_acc': float(sample_acc),
            'fruit_acc': float(f_acc) if f_acc is not None else float(sample_acc),
            'f1': float(f1),
            'precision': float(pre),
            'recall': float(rec),
            'input_feature': 'spatial_mean_224d',
        }
        print(f"    {name:<15s}  Acc={sample_acc:.4f}  F1={f1:.4f}  Fruit={results[name]['fruit_acc']:.4f}")

    return results


# ===========================================================
# 主流程
# ===========================================================
def main():
    print("=" * 70)
    print("  多种子全面实验 — 竞赛报告数据支撑")
    print(f"  时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  种子: {SEEDS}")
    print(f"  数据: {DATA_PATH}")
    print("=" * 70)

    if not os.path.exists(DATA_PATH):
        print(f"错误: 找不到数据文件 {DATA_PATH}")
        return

    d = np.load(DATA_PATH, allow_pickle=True)
    X_train = np.clip(np.nan_to_num(d['X_train_vis']), -10, 10)
    X_val   = np.clip(np.nan_to_num(d['X_val_vis']),   -10, 10)
    X_test  = np.clip(np.nan_to_num(d['X_test_vis']),  -10, 10)
    y_train_raw, y_val_raw, y_test_raw = d['y_train_vis'], d['y_val_vis'], d['y_test_vis']
    test_fruit_ids = d['test_fruit_ids']

    le = LabelEncoder()
    y_tr = le.fit_transform(y_train_raw)
    y_v  = le.transform(y_val_raw)
    y_te = le.transform(y_test_raw)

    print(f"\n数据集: train={len(y_tr)}, val={len(y_v)}, test={len(y_te)}")
    print(f"波段数: {X_train.shape[1]}, 空间: {X_train.shape[2]}x{X_train.shape[3]}")
    print(f"类别: {list(le.classes_)}")
    print(f"测试集果实数: {len(set(test_fruit_ids))}")
    print(f"训练集类别分布: {dict(zip(le.classes_, np.bincount(y_tr)))}")
    print(f"测试集类别分布: {dict(zip(le.classes_, np.bincount(y_te)))}")

    all_results = {
        'meta': {
            'date': datetime.now().isoformat(),
            'seeds': SEEDS,
            'train_samples': len(y_tr),
            'val_samples': len(y_v),
            'test_samples': len(y_te),
            'test_fruits': len(set(test_fruit_ids)),
            'classes': list(le.classes_),
        },
        'ablation': {},
        'comparison_ml': {},
        'comparison_dl': {},
    }

    # ===========================================================
    # Part 1: 消融实验 (4种配置)
    # ===========================================================
    ablation_configs = [
        {'name': 'Baseline',    'use_se': False, 'use_ema': False, 'use_tta': False,
         'use_label_smooth': False, 'use_aug': False, 'use_warmup': False},
        {'name': '+SE',         'use_se': True,  'use_ema': False, 'use_tta': False,
         'use_label_smooth': False, 'use_aug': False, 'use_warmup': False},
        {'name': '+SE+EMA',     'use_se': True,  'use_ema': True,  'use_tta': False,
         'use_label_smooth': True,  'use_aug': True,  'use_warmup': True},
        {'name': 'Full(SE+EMA+TTA)', 'use_se': True, 'use_ema': True, 'use_tta': True,
         'use_label_smooth': True,  'use_aug': True,  'use_warmup': True},
    ]

    print("\n" + "=" * 70)
    print("  Part 1: 消融实验")
    print("=" * 70)

    for cfg in ablation_configs:
        name = cfg['name']
        all_results['ablation'][name] = {'runs': [], 'summary': {}}
        print(f"\n  配置: {name}")
        print(f"    SE={cfg['use_se']}  EMA={cfg['use_ema']}  TTA={cfg['use_tta']}  "
              f"LS={cfg['use_label_smooth']}  Aug={cfg['use_aug']}  Warmup={cfg['use_warmup']}")

        for seed in SEEDS:
            print(f"    Seed={seed} ...", end='', flush=True)
            result = train_one_config(
                X_train, y_tr, X_val, y_v, X_test, y_te, test_fruit_ids, le, seed,
                use_se=cfg['use_se'], use_ema=cfg['use_ema'], use_tta=cfg['use_tta'],
                use_label_smooth=cfg['use_label_smooth'], use_aug=cfg['use_aug'],
                use_warmup=cfg['use_warmup']
            )
            all_results['ablation'][name]['runs'].append(result)
            print(f" Acc={result['sample_acc']:.4f}  Fruit={result['fruit_acc']:.4f}  "
                  f"F1={result['f1']:.4f}  Epochs={result['epochs']}")

        # 汇总 mean ± std
        runs = all_results['ablation'][name]['runs']
        for metric in ['sample_acc', 'fruit_acc', 'f1', 'precision', 'recall']:
            vals = [r[metric] for r in runs]
            all_results['ablation'][name]['summary'][metric] = {
                'mean': float(np.mean(vals)),
                'std': float(np.std(vals)),
                'values': vals,
            }
        all_results['ablation'][name]['summary']['params'] = runs[0]['params']

    # ===========================================================
    # Part 2: 经典 ML 对比
    # ===========================================================
    print("\n" + "=" * 70)
    print("  Part 2: 经典ML对比实验")
    print("=" * 70)

    ml_methods = ['KNN', 'SVM_RBF', 'RandomForest']
    for m in ml_methods:
        all_results['comparison_ml'][m] = {'runs': [], 'summary': {}}

    for seed in SEEDS:
        print(f"\n  Seed={seed}:")
        ml_results = run_ml_comparison(X_train, y_tr, X_test, y_te, test_fruit_ids, le, seed)
        for m, r in ml_results.items():
            all_results['comparison_ml'][m]['runs'].append(r)

    for m in ml_methods:
        runs = all_results['comparison_ml'][m]['runs']
        for metric in ['sample_acc', 'fruit_acc', 'f1', 'precision', 'recall']:
            vals = [r[metric] for r in runs]
            all_results['comparison_ml'][m]['summary'][metric] = {
                'mean': float(np.mean(vals)),
                'std': float(np.std(vals)),
                'values': vals,
            }
        all_results['comparison_ml'][m]['summary']['input_feature'] = 'spatial_mean_224d'

    # ===========================================================
    # Part 3: MLP 对比 (需要单独训练)
    # ===========================================================
    print("\n" + "=" * 70)
    print("  Part 3: MLP对比实验")
    print("=" * 70)

    all_results['comparison_dl']['MLP'] = {'runs': [], 'summary': {}}

    for seed in SEEDS:
        print(f"  Seed={seed} ...", end='', flush=True)
        set_seed(seed)

        # MLP: 空间均值 -> 224维 -> MLP
        X_train_flat = torch.FloatTensor(X_train.mean(axis=(2, 3)))
        X_val_flat   = torch.FloatTensor(X_val.mean(axis=(2, 3)))
        X_test_flat  = torch.FloatTensor(X_test.mean(axis=(2, 3)))

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        class MLPNet(nn.Module):
            def __init__(self, inp, out):
                super().__init__()
                self.net = nn.Sequential(
                    nn.Linear(inp, 512), nn.BatchNorm1d(512), nn.ReLU(True), nn.Dropout(0.5),
                    nn.Linear(512, 256), nn.BatchNorm1d(256), nn.ReLU(True), nn.Dropout(0.5),
                    nn.Linear(256, 128), nn.BatchNorm1d(128), nn.ReLU(True), nn.Dropout(0.5),
                    nn.Linear(128, out),
                )
            def forward(self, x): return self.net(x)

        mlp = MLPNet(224, len(le.classes_)).to(device)
        mlp_opt = optim.AdamW(mlp.parameters(), lr=1e-3, weight_decay=1e-4)
        mlp_crit = nn.CrossEntropyLoss()
        mlp_sched = optim.lr_scheduler.ReduceLROnPlateau(mlp_opt, patience=10, factor=0.5)

        train_ld = DataLoader(TensorDataset(X_train_flat, torch.LongTensor(y_tr)), batch_size=32, shuffle=True)
        val_ld   = DataLoader(TensorDataset(X_val_flat, torch.LongTensor(y_v)),    batch_size=32, shuffle=False)

        best_vl, best_sd, pat = float('inf'), None, 0
        for ep in range(EPOCHS):
            mlp.train()
            for xb, yb in train_ld:
                xb, yb = xb.to(device), yb.to(device)
                mlp_opt.zero_grad(); mlp_crit(mlp(xb), yb).backward(); mlp_opt.step()
            mlp.eval()
            with torch.no_grad():
                vl = sum(mlp_crit(mlp(xb.to(device)), yb.to(device)).item() for xb, yb in val_ld) / len(val_ld)
            mlp_sched.step(vl)
            if vl < best_vl:
                best_vl, best_sd, pat = vl, copy.deepcopy(mlp.state_dict()), 0
            else:
                pat += 1
            if pat >= PATIENCE: break

        mlp.load_state_dict(best_sd); mlp.eval()
        with torch.no_grad():
            probs = F.softmax(mlp(X_test_flat.to(device)), dim=1).cpu().numpy()
        preds = probs.argmax(1)

        s_acc = accuracy_score(y_te, preds)
        f_acc = fruit_level_acc(probs, test_fruit_ids, y_te)
        f1_s  = f1_score(y_te, preds, average='weighted')
        pre_s = precision_score(y_te, preds, average='weighted', zero_division=0)
        rec_s = recall_score(y_te, preds, average='weighted', zero_division=0)

        r = {'seed': seed, 'sample_acc': float(s_acc), 'fruit_acc': float(f_acc),
             'f1': float(f1_s), 'precision': float(pre_s), 'recall': float(rec_s),
             'input_feature': 'spatial_mean_224d', 'hidden_dims': [512, 256, 128]}
        all_results['comparison_dl']['MLP']['runs'].append(r)
        print(f" Acc={s_acc:.4f}  F1={f1_s:.4f}  Fruit={f_acc:.4f}")

    runs = all_results['comparison_dl']['MLP']['runs']
    for metric in ['sample_acc', 'fruit_acc', 'f1', 'precision', 'recall']:
        vals = [r[metric] for r in runs]
        all_results['comparison_dl']['MLP']['summary'][metric] = {
            'mean': float(np.mean(vals)), 'std': float(np.std(vals)), 'values': vals}

    # ===========================================================
    # 打印汇总表
    # ===========================================================
    print("\n\n" + "=" * 80)
    print("  最终结果汇总 (mean ± std, 3种子)")
    print("=" * 80)

    print("\n  --- 消融实验 ---")
    print(f"  {'配置':<25s} {'SampleAcc':>15s} {'FruitAcc':>15s} {'F1':>15s} {'参数量':>10s}")
    print(f"  {'-'*80}")
    for name in ['Baseline', '+SE', '+SE+EMA', 'Full(SE+EMA+TTA)']:
        s = all_results['ablation'][name]['summary']
        sa = s['sample_acc']; fa = s['fruit_acc']; f1s = s['f1']
        p = s['params']
        print(f"  {name:<25s} {sa['mean']:.4f}±{sa['std']:.4f}   "
              f"{fa['mean']:.4f}±{fa['std']:.4f}   "
              f"{f1s['mean']:.4f}±{f1s['std']:.4f}   {p:>8,}")

    print(f"\n  --- 对比实验 ---")
    print(f"  {'方法':<25s} {'SampleAcc':>15s} {'FruitAcc':>15s} {'F1':>15s}")
    print(f"  {'-'*70}")
    for name in ['KNN', 'SVM_RBF', 'RandomForest']:
        s = all_results['comparison_ml'][name]['summary']
        sa = s['sample_acc']; fa = s['fruit_acc']; f1s = s['f1']
        print(f"  {name:<25s} {sa['mean']:.4f}±{sa['std']:.4f}   "
              f"{fa['mean']:.4f}±{fa['std']:.4f}   "
              f"{f1s['mean']:.4f}±{f1s['std']:.4f}")

    s = all_results['comparison_dl']['MLP']['summary']
    sa = s['sample_acc']; fa = s['fruit_acc']; f1s = s['f1']
    print(f"  {'MLP [512,256,128]':<25s} {sa['mean']:.4f}±{sa['std']:.4f}   "
          f"{fa['mean']:.4f}±{fa['std']:.4f}   "
          f"{f1s['mean']:.4f}±{f1s['std']:.4f}")

    # Full model
    s = all_results['ablation']['Full(SE+EMA+TTA)']['summary']
    sa = s['sample_acc']; fa = s['fruit_acc']; f1s = s['f1']
    print(f"  {'HS-CNN+SE (本文)':<25s} {sa['mean']:.4f}±{sa['std']:.4f}   "
          f"{fa['mean']:.4f}±{fa['std']:.4f}   "
          f"{f1s['mean']:.4f}±{f1s['std']:.4f}")

    # 保存结果
    out_path = os.path.join(OUTPUT_DIR, 'multi_seed_results.json')
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\n  结果已保存: {out_path}")
    print("=" * 80)


if __name__ == '__main__':
    main()
