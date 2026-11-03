"""
论文复现 — 原论文设定：随机划分 + 时间假设 + FocalLoss + SGD + 增强
对比当前设定：硬编码划分 + CE + AdamW
"""
import os, json, copy, random, math
import numpy as np
import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from collections import defaultdict

SEED = 23312323
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

BATCH_SIZE = 64; EPOCHS = 500; PATIENCE = 50

# ===== Focal Loss（论文关键组件）=====
class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=None):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
    def forward(self, x, y):
        ce = nn.functional.cross_entropy(x, y, reduction='none')
        pt = torch.exp(-ce)
        focal = ((1 - pt) ** self.gamma) * ce
        if self.alpha is not None:
            focal = self.alpha[y] * focal
        return focal.mean()

# ===== SE Block =====
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
        self.block1 = nn.Sequential(
            nn.Conv2d(bands, bands*k, 7, padding=3, groups=bands),
            nn.Conv2d(bands*k, hidden[0], 1), nn.ReLU(True), nn.AvgPool2d(2),
            nn.BatchNorm2d(hidden[0]), SEBlock(hidden[0]))
        self.block2 = nn.Sequential(
            nn.Conv2d(hidden[0], hidden[0]*k, 3, padding=1, groups=hidden[0]),
            nn.Conv2d(hidden[0]*k, hidden[1], 1), nn.ReLU(True), nn.AvgPool2d(2),
            nn.BatchNorm2d(hidden[1]), SEBlock(hidden[1]))
        self.block3 = nn.Sequential(
            nn.Conv2d(hidden[1], hidden[1]*k, 3, padding=1, groups=hidden[1]),
            nn.Conv2d(hidden[1]*k, hidden[2], 1), nn.ReLU(True), nn.BatchNorm2d(hidden[2]))
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(hidden[2], hidden[2]), nn.BatchNorm1d(hidden[2]),
            nn.ReLU(True), nn.Dropout(0.3), nn.Linear(hidden[2], num_classes))
    def forward(self, x):
        return self.fc(self.gap(self.block3(self.block2(self.block1(x)))).flatten(1))

class EMA:
    def __init__(self, model, decay=0.9):
        self.decay = decay; self.shadow = copy.deepcopy(model).eval()
        for p in self.shadow.parameters(): p.requires_grad_(False)
    def update(self, model):
        for s_p, m_p in zip(self.shadow.parameters(), model.parameters()):
            s_p.data.mul_(self.decay).add_(m_p.data, alpha=1-self.decay)
        for s_b, m_b in zip(self.shadow.buffers(), model.buffers()): s_b.copy_(m_b)

def evaluate(model, loader, device):
    model.eval(); correct, total = 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            correct += (model(x).argmax(1) == y).sum().item(); total += y.size(0)
    return correct / total

def fruit_level_acc(probs, fruit_ids, y):
    d = defaultdict(list); labels = {}
    for i, fid in enumerate(fruit_ids): d[fid].append(probs[i]); labels[fid] = y[i]
    p = [np.stack(v).mean(0).argmax() for v in d.values()]
    return accuracy_score(list(labels.values()), p)

# ===== 数据增强（论文设定：随机翻转+旋转+遮挡）=====
def augment(x):
    """对单个样本应用数据增强"""
    if random.random() > 0.5: x = torch.flip(x, [1])  # 水平翻转
    if random.random() > 0.5:                         # 随机旋转90度
        k = random.randint(1, 3)
        x = torch.rot90(x, k, [1, 2])
    if random.random() > 0.5:                         # 随机遮挡（cutout）
        h, w = x.shape[1], x.shape[2]
        ch, cw = h//4, w//4
        y0 = random.randint(0, h-ch)
        x0 = random.randint(0, w-cw)
        x[:, y0:y0+ch, x0:x0+cw] = 0
    return x


# ===== 用论文的随机划分生成数据 =====
print("用论文的随机划分生成数据...")
import sys
sys.path.insert(0, r'e:\DeepHS-Fruit\deephs_fruit')
from core.datasets.hyperspectral_dataset import get_records
from core.name_convention import *
import spectral

train_records, val_records, test_records = get_records(
    Fruit.AVOCADO, CameraType.VIS, ClassificationType.RIPENESS,
    use_inter_ripeness_levels=True,
    extend_by_time_assumption=True,  # 论文用了时间假设
    random_split=True, seed=23312323)

print(f'Train: {len(train_records)}, Val: {len(val_records)}, Test: {len(test_records)}')

def load_split(records, data_path):
    X, y, fids = [], [], []
    for r in records:
        try:
            p = r.get_file_path()
            full_hdr = os.path.join(data_path, p + '.hdr')
            full_bin = os.path.join(data_path, p + '.bin')
            if not os.path.exists(full_hdr):
                full_hdr = os.path.join(data_path, p + '.hdr')
                alt_p = p.replace('Avocado', 'avocado')
                full_hdr2 = os.path.join(data_path, alt_p + '.hdr')
                if os.path.exists(full_hdr2):
                    full_hdr = full_hdr2
                    full_bin = os.path.join(data_path, alt_p + '.bin')
            img = spectral.envi.open(full_hdr, image=full_bin).load()
            img = np.nan_to_num(img, nan=0.0, posinf=0.0, neginf=0.0)
            img = np.clip(img, -1e10, 1e10)
            t = torch.tensor(img, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
            t = nn.functional.interpolate(t, size=(62,62), mode='bilinear', align_corners=False).squeeze(0)
            padded = np.zeros((224, 64, 64), dtype=np.float32)
            padded[:, 1:-1, 1:-1] = t.numpy()
            label = r.label.ripeness_state
            label_map = {RipenessState.UNRIPE: 'unripe', RipenessState.PERFECT: 'perfect', RipenessState.OVERRIPE: 'overripe'}
            y.append(label_map[label])
            X.append(padded)
            fids.append(str(r.id))
        except Exception as e:
            continue
    return np.array(X), np.array(y), np.array(fids)

data_path = r'e:\DeepHS-Fruit\deephs_fruit\data'
print("\n加载训练集...")
X_train, y_train, _ = load_split(train_records, data_path)
print(f"加载验证集...")
X_val, y_val, _ = load_split(val_records, data_path)
print(f"加载测试集...")
X_test, y_test, test_fruit_ids = load_split(test_records, data_path)

print(f'\n{"="*50}')
print(f'Train: {X_train.shape}  labels={dict(zip(*np.unique(y_train, return_counts=True)))}')
print(f'Val:   {X_val.shape}  labels={dict(zip(*np.unique(y_val, return_counts=True)))}')
print(f'Test:  {X_test.shape}  labels={dict(zip(*np.unique(y_test, return_counts=True)))}')
print(f'{"="*50}')

# 归一化
train_mean = np.mean(X_train, axis=(0,2,3), keepdims=True)
train_std = np.std(X_train, axis=(0,2,3), keepdims=True)
train_std[train_std < 1e-6] = 1
for arr in [X_train, X_val, X_test]:
    arr[:] = (arr - train_mean) / train_std
    np.clip(arr, -10, 10, out=arr)

X_train = torch.FloatTensor(X_train)
X_val = torch.FloatTensor(X_val)
X_test = torch.FloatTensor(X_test)

le = LabelEncoder()
y_tr = le.fit_transform(y_train)
y_v = le.transform(y_val)
y_te = le.transform(y_test)
n_classes = len(le.classes_)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
bands = X_train.shape[1]
print(f'Device: {device}, Classes: {n_classes}')

# ===== 训练 =====
def train_paper_style(name, use_focal=True, use_aug=True, use_sgd=True, use_ema=True, use_tta=True):
    print(f'\n{"="*50}')
    print(f'配置: {name}')
    print(f'  focal={use_focal}, aug={use_aug}, sgd={use_sgd}, ema={use_ema}, tta={use_tta}')
    print(f'{"="*50}')

    model = HSClassifier(bands=bands, num_classes=n_classes).to(device)
    
    if use_sgd:
        opt = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.StepLR(opt, step_size=10, gamma=0.5)
    else:
        opt = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        scheduler = None

    criterion = FocalLoss(gamma=2.0) if use_focal else nn.CrossEntropyLoss()
    ema_obj = EMA(model, decay=0.9) if use_ema else None

    tl = DataLoader(TensorDataset(X_train, torch.LongTensor(y_tr)), BATCH_SIZE, shuffle=True)
    vl = DataLoader(TensorDataset(X_val, torch.LongTensor(y_v)), BATCH_SIZE)
    tel = DataLoader(TensorDataset(X_test, torch.LongTensor(y_te)), BATCH_SIZE)

    best_acc = 0; best_sd = None; best_ema_sd = None; patience = 0
    for ep in range(EPOCHS):
        model.train()
        for x, y in tl:
            if use_aug:
                x = torch.stack([augment(xi) for xi in x])
            x, y = x.to(device), y.to(device)
            opt.zero_grad(); criterion(model(x), y).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            if ema_obj: ema_obj.update(model)
        if scheduler: scheduler.step()

        if ema_obj: acc = evaluate(ema_obj.shadow, vl, device)
        else: acc = evaluate(model, vl, device)

        if acc > best_acc:
            best_acc = acc
            best_sd = copy.deepcopy(model.state_dict())
            if ema_obj: best_ema_sd = copy.deepcopy(ema_obj.shadow.state_dict())
            patience = 0
        else: patience += 1
        if patience >= PATIENCE:
            print(f'  Early stop at epoch {ep+1}, best_val={best_acc*100:.1f}%')
            break

    if ema_obj and best_ema_sd is not None:
        ema_obj.shadow.load_state_dict(best_ema_sd); inf_model = ema_obj.shadow
    else:
        model.load_state_dict(best_sd); inf_model = model
    inf_model.eval()

    # TTA
    if use_tta:
        all_probs = []
        for _ in range(5):
            preds = []
            with torch.no_grad():
                for x, _ in tel:
                    x = x.to(device); xi = x.clone()
                    for i in range(len(xi)):
                        if torch.rand(1) > 0.5: xi[i] = torch.flip(xi[i], [1])
                        if torch.rand(1) > 0.5: xi[i] = torch.flip(xi[i], [2])
                        if torch.rand(1) > 0.5: xi[i] = torch.rot90(xi[i], 1, [1, 2])
                    preds.append(torch.softmax(inf_model(xi), 1).cpu())
            all_probs.append(torch.cat(preds))
        probs = torch.stack(all_probs).mean(0).numpy()
    else:
        with torch.no_grad():
            probs = torch.cat([torch.softmax(inf_model(x.to(device)), 1).cpu() for x, _ in tel]).numpy()

    sa = accuracy_score(y_te, probs.argmax(1))
    fa = fruit_level_acc(probs, test_fruit_ids, y_te)
    print(f'  样本准确率: {sa*100:.1f}%  水果准确率: {fa*100:.1f}%')
    return sa, fa

# ===== 跑4组对比 =====
results = []
for name, focal, aug, sgd, ema, tta in [
    ('论文完整配置', True, True, True, True, True),
    ('无Focal',     False, True, True, True, True),
    ('无增强',      True, False, True, True, True),
    ('无SGD(AdamW)', True, True, False, True, True),
]:
    sa, fa = train_paper_style(name, focal, aug, sgd, ema, tta)
    results.append({'name': name, 'sample_acc': round(float(sa),4), 'fruit_acc': round(float(fa),4)})

print(f'\n{"="*60}')
print('论文复现结果汇总')
print(f'{"="*60}')
print(f'{"配置":20s}  {"样本准确率":12s}  {"水果准确率":12s}')
print(f'{"-"*50}')
for r in results:
    print(f'{r["name"]:20s}  {r["sample_acc"]*100:6.1f}%      {r["fruit_acc"]*100:6.1f}%')
print(f'{"="*60}')
