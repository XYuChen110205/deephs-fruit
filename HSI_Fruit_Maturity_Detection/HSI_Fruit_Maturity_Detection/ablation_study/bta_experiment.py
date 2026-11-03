"""
双向时间假设 (Bidirectional Time Assumption, BTA)
===================================================
改进原论文的单向扩展：向前推unripe + 向后推overripe + 中间填perfect
"""
import os, sys, copy, random, math, json
import numpy as np
import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from collections import defaultdict, Counter

# ===== 读论文的数据划分 =====
sys.path.insert(0, r'e:\DeepHS-Fruit\deephs_fruit')
from core.datasets.hyperspectral_dataset import get_records
from core.name_convention import *
import spectral

SEED = 23312323
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

BATCH_SIZE = 64; EPOCHS = 300; PATIENCE = 40

# ===== 获取原始记录 =====
train_records, val_records, test_records = get_records(
    Fruit.AVOCADO, CameraType.VIS, ClassificationType.RIPENESS,
    use_inter_ripeness_levels=True,
    extend_by_time_assumption=False,  # 不用原论文的时间假设，我们自己实现
    random_split=True, seed=23312323)

print(f'原始: Train={len(train_records)}, Val={len(val_records)}, Test={len(test_records)}')

# ===== 双向时间假设扩展 =====
def apply_bta(records):
    """
    对每个水果按时间排序，应用双向时间假设：
    1. 已知某天是 unripe → 之前所有天都是 unripe
    2. 已知某天是 overripe → 之后所有天都是 overripe
    3. unripe 之后、overripe 之前 → perfect
    """
    # 按水果ID分组
    id_to_records = defaultdict(list)
    for r in records:
        id_to_records[r.id].append(r)
    
    # 天数排序
    day_order = [
        'day_01','day_02','day_03','day_04','day_05','day_06','day_07',
        'day_08','day_09','day_10','day_m2_01','day_m2_02','day_m2_03',
        'day_m2_04','day_m2_05','day_m2_06','day_m2_07','day_m2_08',
        'day_m2_09','day_m2_10','day_m2_11','day_m2_12','day_m2_13',
        'day_m2_14','day_m4_01','day_m4_02','day_m4_03','day_m4_04',
        'day_m4_05','day_m4_06','day_m4_07','day_m4_08','day_m4_09'
    ]
    day_rank = {d: i for i, d in enumerate(day_order)}

    extended = []
    for fid, recs in id_to_records.items():
        recs.sort(key=lambda r: day_rank.get(r.day.value, 999))

        labeled = [(i, r) for i, r in enumerate(recs) if r.is_labeled()]
        if not labeled:
            extended.extend(recs)
            continue

        # 找到第一个 unripe 和最后一个 overripe 的位置
        first_unripe_idx = None
        last_overripe_idx = None
        for i, r in enumerate(recs):
            if r.is_labeled() and r.label.ripeness_state == RipenessState.UNRIPE:
                if first_unripe_idx is None: first_unripe_idx = i
            if r.is_labeled() and r.label.ripeness_state == RipenessState.OVERRIPE:
                last_overripe_idx = i

        # 应用时间假设
        for i, r in enumerate(recs):
            if not r.is_labeled():
                if first_unripe_idx is not None and i < first_unripe_idx:
                    r.label = copy.copy(labeled[0][1].label)
                    r.label.ripeness_state = RipenessState.UNRIPE
                    r.label.comment = "BTA: unripe extended backward"
                elif last_overripe_idx is not None and i > last_overripe_idx:
                    # 找一个overripe标签来复制
                    for _, lr in labeled:
                        if lr.label.ripeness_state == RipenessState.OVERRIPE:
                            r.label = copy.copy(lr.label)
                            r.label.ripeness_state = RipenessState.OVERRIPE
                            r.label.comment = "BTA: overripe extended forward"
                            break
                elif first_unripe_idx is not None and last_overripe_idx is not None:
                    if i > first_unripe_idx and i < last_overripe_idx:
                        for _, lr in labeled:
                            if lr.label.ripeness_state == RipenessState.PERFECT:
                                r.label = copy.copy(lr.label)
                                r.label.ripeness_state = RipenessState.PERFECT
                                r.label.comment = "BTA: perfect interpolated"
                                break
            extended.append(r)
    
    return extended

# ===== 加载数据 =====
data_path = r'e:\DeepHS-Fruit\deephs_fruit\data'

def load_split(records, data_path):
    X, y, fids = [], [], []
    for r in records:
        try:
            p = r.get_file_path()
            full_hdr = os.path.join(data_path, p + '.hdr')
            full_bin = os.path.join(data_path, p + '.bin')
            if not os.path.exists(full_hdr):
                alt = p.replace('Avocado', 'avocado')
                full_hdr = os.path.join(data_path, alt + '.hdr')
                full_bin = os.path.join(data_path, alt + '.bin')
            img = spectral.envi.open(full_hdr, image=full_bin).load()
            img = np.nan_to_num(img, nan=0.0, posinf=0.0, neginf=0.0)
            img = np.clip(img, -1e10, 1e10)
            t = torch.tensor(img, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
            t = nn.functional.interpolate(t, size=(62,62), mode='bilinear', align_corners=False).squeeze(0)
            padded = np.zeros((224, 64, 64), dtype=np.float32)
            padded[:, 1:-1, 1:-1] = t.numpy()
            label = r.label.ripeness_state
            lm = {RipenessState.UNRIPE: 'unripe', RipenessState.PERFECT: 'perfect', RipenessState.OVERRIPE: 'overripe'}
            y.append(lm[label])
            X.append(padded)
            fids.append(str(r.id))
        except Exception as e:
            continue
    return np.array(X), np.array(y), np.array(fids)

# ==== 原始划分（无时间假设）====
print("\n加载原始训练集（无扩展）...")
X_tr_orig, y_tr_orig, _ = load_split(train_records, data_path)
X_val, y_val, _ = load_split(val_records, data_path)
X_te, y_te, te_fids = load_split(test_records, data_path)

# ==== 应用 BTA 后再加载 ====
print("应用双向时间假设扩展...")
train_extended = apply_bta(train_records)
print(f"BTA扩展: {len(train_records)} -> {len(train_extended)} 条记录")
for r in train_extended:
    if r.label and r.label.comment and 'BTA' in r.label.comment:
        pass  # 只是确认扩展生效
labeled_cnt = sum(1 for r in train_extended if r.is_labeled())
print(f"  其中有标注的: {labeled_cnt}")

print("\n加载 BTA 扩展后的训练集...")
X_tr_bta, y_tr_bta, _ = load_split(train_extended, data_path)

def print_stats(name, X, y):
    cnt = Counter(y)
    print(f'{name}: {len(X)} 张  labels={dict(cnt)}')
    if len(cnt) > 1:
        total = sum(cnt.values())
        for k, v in sorted(cnt.items()):
            print(f'    {k}: {v} ({v/total*100:.0f}%)')

print(f'\n{"="*50}')
print_stats('原始训练集', X_tr_orig, y_tr_orig)
print_stats('BTA扩展训练集', X_tr_bta, y_tr_bta)
print_stats('验证集', X_val, y_val)
print_stats('测试集', X_te, y_te)
print(f'{"="*50}')

# ==== 归一化 ====
def normalize(X_train, X_val, X_test):
    mean = np.mean(X_train, axis=(0,2,3), keepdims=True)
    std = np.std(X_train, axis=(0,2,3), keepdims=True)
    std[std < 1e-6] = 1
    for arr in [X_train, X_val, X_test]:
        arr[:] = (arr - mean) / std
        np.clip(arr, -10, 10, out=arr)
    return X_train, X_val, X_test

X_tr_orig, X_val, X_te = normalize(X_tr_orig.copy(), X_val.copy(), X_te.copy())
X_tr_bta_t, X_val2, X_te2 = normalize(X_tr_bta.copy(), X_val.copy(), X_te.copy())

# ===== 模型 =====
class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        mid = max(channels // reduction, 4)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(nn.Linear(channels, mid), nn.ReLU(True), nn.Linear(mid, channels), nn.Sigmoid())
    def forward(self, x):
        w = self.fc(self.gap(x).flatten(1)).unsqueeze(-1).unsqueeze(-1)
        return x * w

class HSClassifier(nn.Module):
    def __init__(self, bands=224, num_classes=3, hidden=[64, 128, 256]):
        super().__init__()
        k = 3
        self.block1 = nn.Sequential(nn.Conv2d(bands, bands*k, 7, padding=3, groups=bands), nn.Conv2d(bands*k, hidden[0], 1), nn.ReLU(True), nn.AvgPool2d(2), nn.BatchNorm2d(hidden[0]), SEBlock(hidden[0]))
        self.block2 = nn.Sequential(nn.Conv2d(hidden[0], hidden[0]*k, 3, padding=1, groups=hidden[0]), nn.Conv2d(hidden[0]*k, hidden[1], 1), nn.ReLU(True), nn.AvgPool2d(2), nn.BatchNorm2d(hidden[1]), SEBlock(hidden[1]))
        self.block3 = nn.Sequential(nn.Conv2d(hidden[1], hidden[1]*k, 3, padding=1, groups=hidden[1]), nn.Conv2d(hidden[1]*k, hidden[2], 1), nn.ReLU(True), nn.BatchNorm2d(hidden[2]))
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(nn.Linear(hidden[2], hidden[2]), nn.BatchNorm1d(hidden[2]), nn.ReLU(True), nn.Dropout(0.3), nn.Linear(hidden[2], num_classes))
    def forward(self, x):
        return self.fc(self.gap(self.block3(self.block2(self.block1(x)))).flatten(1))

class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0):
        super().__init__()
        self.gamma = gamma
    def forward(self, x, y):
        ce = nn.functional.cross_entropy(x, y, reduction='none')
        return (((1 - torch.exp(-ce)) ** self.gamma) * ce).mean()

class EMA:
    def __init__(self, model, decay=0.9):
        self.decay = decay; self.shadow = copy.deepcopy(model).eval()
        for p in self.shadow.parameters(): p.requires_grad_(False)
    def update(self, model):
        for s_p, m_p in zip(self.shadow.parameters(), model.parameters()):
            s_p.data.mul_(self.decay).add_(m_p.data, alpha=1-self.decay)
        for s_b, m_b in zip(self.shadow.buffers(), model.buffers()): s_b.copy_(m_b)

def augment(x):
    if random.random() > 0.5: x = torch.flip(x, [1])
    if random.random() > 0.5:
        k = random.randint(1, 3)
        x = torch.rot90(x, k, [1, 2])
    if random.random() > 0.5:
        h, w = x.shape[1], x.shape[2]
        ch, cw = h//4, w//4
        y0 = random.randint(0, h-ch); x0 = random.randint(0, w-cw)
        x[:, y0:y0+ch, x0:x0+cw] = 0
    return x

def evaluate(model, loader, device):
    model.eval(); correct, total = 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            correct += (model(x).argmax(1) == y).sum().item(); total += y.size(0)
    return correct / total

def train_model(X_train, y_train, name, use_aug=True):
    print(f'\n{"="*50}')
    print(f'训练: {name}')
    
    le = LabelEncoder()
    y_tr = le.fit_transform(y_train)
    y_v = le.transform(y_val)
    y_te_t = le.transform(y_te)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    n_classes = len(le.classes_)

    X_tr_t = torch.FloatTensor(X_train)
    X_v_t = torch.FloatTensor(X_val)
    X_te_t = torch.FloatTensor(X_te)

    tl = DataLoader(TensorDataset(X_tr_t, torch.LongTensor(y_tr)), BATCH_SIZE, shuffle=True)
    vl = DataLoader(TensorDataset(X_v_t, torch.LongTensor(y_v)), BATCH_SIZE)
    tel = DataLoader(TensorDataset(X_te_t, torch.LongTensor(y_te_t)), BATCH_SIZE)

    model = HSClassifier(bands=224, num_classes=n_classes).to(device)
    opt = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(opt, step_size=10, gamma=0.5)
    criterion = FocalLoss(gamma=2.0)
    ema_obj = EMA(model, decay=0.9)

    best_acc = 0; best_sd = None; best_ema_sd = None; patience = 0
    for ep in range(EPOCHS):
        model.train()
        for x, y in tl:
            if use_aug:
                x = torch.stack([augment(xi) for xi in x])
            x, y = x.to(device), y.to(device)
            opt.zero_grad(); criterion(model(x), y).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); ema_obj.update(model)
        scheduler.step()
        acc = evaluate(ema_obj.shadow, vl, device)
        if acc > best_acc:
            best_acc = acc; best_sd = copy.deepcopy(model.state_dict())
            best_ema_sd = copy.deepcopy(ema_obj.shadow.state_dict()); patience = 0
        else: patience += 1
        if patience >= PATIENCE:
            print(f'  Early stop at epoch {ep+1}, val_acc={best_acc*100:.1f}%')
            break

    ema_obj.shadow.load_state_dict(best_ema_sd)
    model = ema_obj.shadow; model.eval()

    with torch.no_grad():
        probs = torch.cat([torch.softmax(model(x.to(device)), 1).cpu() for x, _ in tel]).numpy()
    
    preds = probs.argmax(1)
    sa = accuracy_score(y_te_t, preds)
    cm = confusion_matrix(y_te_t, preds)
    print(f'  测试准确率: {sa*100:.1f}%')
    print(f'  混淆矩阵:\n{cm}')
    return sa, cm

results = []
for X_tr, y_tr, name in [
    (X_tr_orig, y_tr_orig, '原始 (92张, 无时间假设)'),
    (X_tr_bta, y_tr_bta, 'BTA扩展 (双向时间假设)'),
]:
    sa, cm = train_model(X_tr, y_tr, name)
    results.append({'name': name, 'acc': round(float(sa),4)})

print(f'\n{"="*60}')
print('对比结果')
print(f'{"="*60}')
for r in results:
    print(f'{r["name"]:35s}  准确率: {r["acc"]*100:.1f}%')
