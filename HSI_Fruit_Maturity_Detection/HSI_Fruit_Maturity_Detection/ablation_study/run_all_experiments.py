"""
========================================================
  自动化批量实验 — HS-CNN 牛油果成熟度消融研究
========================================================
一次性跑 6-8 个配置，结果自动保存到 ablation_comparison_results.json

运行方式（放那去吃饭就行）:
  cd E:\DeepHS-Fruit\HSI_Fruit_Maturity_Detection\HSI_Fruit_Maturity_Detection\ablation_study
  & "E:\miniconda3\envs\hyperspectral\python.exe" run_all_experiments.py

预计时间: 2-4 小时（取决于 GPU）
========================================================
"""
import os, sys, copy, random, json, time
import numpy as np
import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from collections import defaultdict, Counter

# ==================== 配置 ====================
SEED = 23312323
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

BATCH_SIZE = 64
EPOCHS = 500
PATIENCE = 40

OUTPUT_FILE = 'ablation_comparison_results.json'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'使用设备: {device}')
if device.type == 'cuda':
    print(f'  GPU: {torch.cuda.get_device_name(0)}')
    print(f'  显存: {torch.cuda.get_device_properties(0).total_memory/1024**3:.1f} GB')

# ==================== 数据加载 ====================
sys.path.insert(0, r'e:\DeepHS-Fruit\deephs_fruit')
from core.datasets.hyperspectral_dataset import get_records
from core.name_convention import *
import spectral

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

# ====== 1. 硬编码划分（v1）======
print('\n[1/5] 加载硬编码划分...')
records_v1, _, _ = get_records(Fruit.AVOCADO, CameraType.VIS, ClassificationType.RIPENESS,
    use_inter_ripeness_levels=True, extend_by_time_assumption=False, random_split=False)
# 硬编码划分下 train/val/test 是固定的
# 我们需要从 records 里分别加载
val_set = [r for r in records_v1 if hasattr(r, 'get_dataset') and False]  # 跳过，直接硬编码路径

# 实际上用之前的准备工作：直接加载已知路径
# 硬编码划分的 test 是 test_v2.json, 我们需要用完全相同的划分
# 简单起见，我们用之前准备好 ablation_data_v2.npz
# 但这个文件没有保存 labels 的字符串映射

# ====== 更好的方式：直接用之前的 npz 文件 ======
# 训练用 v2 硬编码划分的数据来自之前的准备工作
# 我们直接重新走一遍标准流程

# 为了简化，我们统一用 get_records 的随机划分作为基础
# 然后分别尝试不同配置

print('\n[2/5] 获取论文随机划分...')
train_records, val_records, test_records = get_records(
    Fruit.AVOCADO, CameraType.VIS, ClassificationType.RIPENESS,
    use_inter_ripeness_levels=True, extend_by_time_assumption=False,
    random_split=True, seed=SEED)
print(f'  原始划分: Train={len(train_records)}, Val={len(val_records)}, Test={len(test_records)}')

print('\n[3/5] 应用 BTA 双向时间假设...')

def apply_bta(records):
    """双向时间假设：向前推unripe + 向后推overripe + 中间perfect"""
    id_to_records = defaultdict(list)
    for r in records:
        id_to_records[r.id].append(r)
    
    day_order = ['day_01','day_02','day_03','day_04','day_05','day_06','day_07',
        'day_08','day_09','day_10','day_m2_01','day_m2_02','day_m2_03',
        'day_m2_04','day_m2_05','day_m2_06','day_m2_07','day_m2_08',
        'day_m2_09','day_m2_10','day_m2_11','day_m2_12','day_m2_13',
        'day_m2_14','day_m4_01','day_m4_02','day_m4_03','day_m4_04',
        'day_m4_05','day_m4_06','day_m4_07','day_m4_08','day_m4_09']
    day_rank = {d: i for i, d in enumerate(day_order)}

    extended = []
    for fid, recs in id_to_records.items():
        recs = sorted(recs, key=lambda r: day_rank.get(r.day.value, 999))
        labeled = [r for r in recs if r.is_labeled()]
        if not labeled:
            extended.extend(recs); continue

        first_unripe = None
        last_overripe = None
        a_perfect = None
        for r in labeled:
            if r.label.ripeness_state == RipenessState.UNRIPE and first_unripe is None:
                first_unripe = r
            if r.label.ripeness_state == RipenessState.OVERRIPE:
                last_overripe = r
            if r.label.ripeness_state == RipenessState.PERFECT and a_perfect is None:
                a_perfect = r

        for r in recs:
            if r.is_labeled():
                extended.append(r)
                continue
            if first_unripe and r.day.value < first_unripe.day.value:
                new_label = copy.copy(first_unripe.label)
                new_label.ripeness_state = RipenessState.UNRIPE
                r.label = new_label
            elif last_overripe and r.day.value > last_overripe.day.value:
                new_label = copy.copy(last_overripe.label)
                new_label.ripeness_state = RipenessState.OVERRIPE
                r.label = new_label
            elif first_unripe and last_overripe and a_perfect:
                fu_day = day_rank.get(first_unripe.day.value, 0)
                lo_day = day_rank.get(last_overripe.day.value, 999)
                rd = day_rank.get(r.day.value, 500)
                if fu_day < rd < lo_day:
                    new_label = copy.copy(a_perfect.label)
                    new_label.ripeness_state = RipenessState.PERFECT
                    r.label = new_label
            extended.append(r)
    return extended

# 加载所有数据划分
print('\n[4/5] 加载数据到内存（这步最慢，耐心等）...')

# --- 配置A: 原始随机划分（无时间假设）---
print('  加载原始训练集...')
X_tr_raw, y_tr_raw, _ = load_split(train_records, data_path)

# --- 配置B: 原始论文的时间假设（只推unripe）---
print('  加载原论文单向时间假设...')
sys.path.insert(0, r'e:\DeepHS-Fruit\deephs_fruit')
from core.datasets.hyperspectral_dataset import extend_unripe_recordings
train_paper = list(train_records)
train_paper_extended = extend_unripe_recordings(train_paper)
X_tr_paper, y_tr_paper, _ = load_split(train_paper_extended, data_path)

# --- 配置C: BTA双向时间假设 ---
print('  加载BTA双向时间假设...')
train_bta = apply_bta(train_records)
X_tr_bta, y_tr_bta, _ = load_split(train_bta, data_path)

# --- 验证集和测试集（所有配置共享）---
print('  加载验证集...')
X_val, y_val, _ = load_split(val_records, data_path)
print('  加载测试集...')
X_te, y_te, te_fids = load_split(test_records, data_path)

raw_configs = [
    ('A-原始 (92张)', X_tr_raw, y_tr_raw),
    ('B-论文扩展 (原论文)', X_tr_paper, y_tr_paper),
    ('C-BTA (双向)', X_tr_bta, y_tr_bta),
]

print(f'\n{"="*60}')
print('数据集统计')
print(f'{"="*60}')
for name, X, y in raw_configs:
    cnt = Counter(y)
    print(f'{name:25s}: {len(X):4d}张  {dict(cnt)}')
print(f'{"验证集":25s}: {len(X_val):4d}张  {dict(Counter(y_val))}')
print(f'{"测试集":25s}: {len(X_te):4d}张  {dict(Counter(y_te))}')
print(f'{"="*60}')

# ==================== 归一化 ====================
def normalize(X_train, X_val, X_test):
    mean = np.mean(X_train, axis=(0,2,3), keepdims=True)
    std = np.std(X_train, axis=(0,2,3), keepdims=True)
    std[std < 1e-6] = 1
    X_train_n = (X_train - mean) / std
    X_val_n = (X_val - mean) / std
    X_test_n = (X_test - mean) / std
    for arr in [X_train_n, X_val_n, X_test_n]:
        np.clip(arr, -10, 10, out=arr)
    return X_train_n, X_val_n, X_test_n

# 对每个数据集分别归一化
dataset_configs = []
for name, X, y in raw_configs:
    X_tr_n, X_v_n, X_te_n = normalize(X, X_val.copy(), X_te.copy())
    dataset_configs.append((name, X_tr_n, X_v_n, X_te_n, y))

# ==================== 模型 ====================
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
        super().__init__(); self.gamma = gamma
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
        k = random.randint(1, 3); x = torch.rot90(x, k, [1, 2])
    if random.random() > 0.5:
        h, w = x.shape[1], x.shape[2]
        ch, cw = h//4, w//4
        y0 = random.randint(0, h-ch); x0 = random.randint(0, w-cw)
        x[:, y0:y0+ch, x0:x0+cw] = 0
    return x

def evaluate(model, loader):
    model.eval(); correct, total = 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            correct += (model(x).argmax(1) == y).sum().item(); total += y.size(0)
    return correct / total

# ==================== 训练函数 ====================
def train_one_exp(name, X_train, y_train, X_val, X_test, y_test, use_aug=True, use_focal=True, use_sgd=True):
    le = LabelEncoder()
    y_tr = le.fit_transform(y_train)
    y_v = le.transform(y_val)
    y_te = le.transform(y_test)
    n_classes = len(le.classes_)

    X_tr = torch.FloatTensor(X_train).pin_memory()
    X_v = torch.FloatTensor(X_val).pin_memory()
    X_te_t = torch.FloatTensor(X_test).pin_memory()

    tl = DataLoader(TensorDataset(X_tr, torch.LongTensor(y_tr)), BATCH_SIZE, shuffle=True, pin_memory=True)
    vl = DataLoader(TensorDataset(X_v, torch.LongTensor(y_v)), BATCH_SIZE, pin_memory=True)
    tel = DataLoader(TensorDataset(X_te_t, torch.LongTensor(y_te)), BATCH_SIZE, pin_memory=True)

    model = HSClassifier(bands=224, num_classes=n_classes).to(device)
    
    if use_sgd:
        opt = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.StepLR(opt, step_size=10, gamma=0.5)
    else:
        opt = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        scheduler = None

    criterion = FocalLoss(gamma=2.0) if use_focal else nn.CrossEntropyLoss()
    ema_obj = EMA(model, decay=0.9)

    best_acc = 0; best_ema_sd = None; patience = 0; final_ep = EPOCHS
    for ep in range(EPOCHS):
        model.train()
        for x, y in tl:
            if use_aug:
                x = torch.stack([augment(xi) for xi in x])
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            opt.zero_grad(); criterion(model(x), y).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); ema_obj.update(model)
        if scheduler: scheduler.step()
        acc = evaluate(ema_obj.shadow, vl)
        if acc > best_acc:
            best_acc = acc; best_ema_sd = copy.deepcopy(ema_obj.shadow.state_dict()); patience = 0
        else: patience += 1
        if patience >= PATIENCE:
            final_ep = ep + 1; break

    ema_obj.shadow.load_state_dict(best_ema_sd)
    model = ema_obj.shadow; model.eval()
    
    with torch.no_grad():
        probs = torch.cat([torch.softmax(model(x.to(device)), 1).cpu() for x, _ in tel]).numpy()
    preds = probs.argmax(1)
    
    sa = accuracy_score(y_te, preds)
    f1 = f1_score(y_te, preds, average='weighted') if n_classes > 2 else f1_score(y_te, preds)
    cm = confusion_matrix(y_te, preds).tolist()
    
    result = {
        'name': name, 'epochs': final_ep, 'val_acc': round(float(best_acc), 4),
        'sample_acc': round(float(sa), 4), 'f1': round(float(f1), 4),
        'confusion_matrix': cm, 'n_train': len(X_train),
    }
    print(f'  [{name:30s}] ep={final_ep:3d}  val={best_acc*100:5.1f}%  test={sa*100:5.1f}%  f1={f1*100:.1f}%')
    return result

# ==================== 跑所有实验 ====================
print(f'\n{"="*60}')
print('开始批量训练（去吃饭吧，回来就好了）')
print(f'{"="*60}')

all_results = []
exp_counter = 0

for ds_name, X_tr, X_v, X_te_n, y in dataset_configs:
    for aug in [True, False]:
        for focal in [True, False]:
            exp_counter += 1
            aug_str = '+Aug' if aug else '无Aug'
            focal_str = '+Focal' if focal else '+CE'
            full_name = f'{ds_name} {focal_str} {aug_str}'
            print(f'\n--- 实验 {exp_counter}: {full_name} ---')
            t0 = time.time()
            r = train_one_exp(full_name, X_tr, y, X_v, X_te_n, y_te, use_aug=aug, use_focal=focal)
            r['time'] = round(time.time() - t0)
            all_results.append(r)

# ==================== 结果汇总 ====================
print(f'\n\n{"="*70}')
print(f'全部 {len(all_results)} 个实验完成！')
print(f'{"="*70}')
print(f'{"实验配置":50s}  {"准确率":8s}  {"F1":8s}  {"训练集":6s}  {"耗时":6s}')
print(f'{"-"*70}')
all_results.sort(key=lambda x: x['sample_acc'], reverse=True)
for r in all_results:
    print(f'{r["name"]:50s}  {r["sample_acc"]*100:5.1f}%  {r["f1"]*100:5.1f}%  {r["n_train"]:4d}张  {r["time"]//60:2d}m')
print(f'{"="*70}')
print(f'最佳配置: {all_results[0]["name"]}')
print(f'最佳准确率: {all_results[0]["sample_acc"]*100:.1f}%')

# 保存
output = {
    'device': str(device),
    'seed': SEED,
    'results': all_results,
    'best': all_results[0],
}
with open(OUTPUT_FILE, 'w') as f:
    json.dump(output, f, indent=2, ensure_ascii=False)
print(f'\n结果已保存到: {OUTPUT_FILE}')
print(f'\n{"="*70}')
print('跑完了！用网页看结果:')
print(f'  确保 app.py 在运行')
print(f'  然后访问 http://localhost:5000')
print(f'{"="*70}')
