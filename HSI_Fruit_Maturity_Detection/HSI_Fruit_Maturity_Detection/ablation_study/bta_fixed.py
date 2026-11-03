"""
修复版 BTA 实验 — 直接从完整数据集扩展
核心问题: get_records 只返回有标注的，BTA 必须在完整数据集上做
"""
import os, sys, copy, random, json, time
import numpy as np
import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from collections import defaultdict, Counter

SEED = 23312323
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
BATCH_SIZE = 64; EPOCHS = 300; PATIENCE = 40
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'设备: {device}')

sys.path.insert(0, r'e:\DeepHS-Fruit\deephs_fruit')
from core.name_convention import *
from core.measurements import all_fruits_new
from core import fruit_list
import spectral

data_path = r'e:\DeepHS-Fruit\deephs_fruit\data'

# ===== 获取完整数据集（含无标注）=====
print('\n获取完整 avocado VIS 数据...')
full_set = fruit_list.get_for_camera_type(
    fruit_list.get_for_fruit(all_fruits_new, Fruit.AVOCADO), CameraType.VIS)
print(f'总计: {len(full_set)} 条')
print(f'有标注: {sum(1 for r in full_set if r.is_labeled())}')
print(f'无标注: {sum(1 for r in full_set if not r.is_labeled())}')

# ===== 天数排序表 =====
day_order = ['day_01','day_02','day_03','day_04','day_05','day_06','day_07',
    'day_08','day_09','day_10','day_m2_01','day_m2_02','day_m2_03',
    'day_m2_04','day_m2_05','day_m2_06','day_m2_07','day_m2_08',
    'day_m2_09','day_m2_10','day_m2_11','day_m2_12','day_m2_13',
    'day_m2_14','day_m4_01','day_m4_02','day_m4_03','day_m4_04',
    'day_m4_05','day_m4_06','day_m4_07','day_m4_08','day_m4_09']
day_rank = {d: i for i, d in enumerate(day_order)}

def apply_bta_to_fullset(records):
    """对完整数据集应用 BTA，返回扩展后的列表"""
    id_to_records = defaultdict(list)
    for r in records:
        id_to_records[r.id].append(r)

    extended = []
    for fid, recs in id_to_records.items():
        recs = sorted(recs, key=lambda r: day_rank.get(r.day.value, 999))
        labeled = [r for r in recs if r.is_labeled()]
        if not labeled:
            extended.extend(recs); continue

        # 找到第一个 UNRIPE 和最后一个 OVERRIPE 位置（按天数排序）
        sorted_labeled = sorted(labeled, key=lambda r: day_rank.get(r.day.value, 999))
        first_unripe = None
        last_overripe = None
        perfect_sample = None
        for r in sorted_labeled:
            if r.label.ripeness_state == RipenessState.UNRIPE and first_unripe is None:
                first_unripe = r
            if r.label.ripeness_state == RipenessState.OVERRIPE:
                last_overripe = r
            if r.label.ripeness_state == RipenessState.PERFECT and perfect_sample is None:
                perfect_sample = r

        # 复制标签
        unripe_label = copy.copy(first_unripe.label) if first_unripe else None
        overripe_label = copy.copy(last_overripe.label) if last_overripe else None
        perfect_label = copy.copy(perfect_sample.label) if perfect_sample else None

        for r in recs:
            if r.is_labeled():
                extended.append(r); continue
            rd = day_rank.get(r.day.value, 500)
            if first_unripe and rd < day_rank.get(first_unripe.day.value, 0):
                new_lb = copy.copy(unripe_label)
                new_lb.ripeness_state = RipenessState.UNRIPE
                new_lb.comment = 'BTA: unripe backward'
                r.label = new_lb
            elif last_overripe and rd > day_rank.get(last_overripe.day.value, 999):
                new_lb = copy.copy(overripe_label)
                new_lb.ripeness_state = RipenessState.OVERRIPE
                new_lb.comment = 'BTA: overripe forward'
                r.label = new_lb
            elif first_unripe and last_overripe and perfect_label:
                fu_rank = day_rank.get(first_unripe.day.value, 0)
                lo_rank = day_rank.get(last_overripe.day.value, 999)
                if fu_rank < rd < lo_rank:
                    new_lb = copy.copy(perfect_label)
                    new_lb.ripeness_state = RipenessState.PERFECT
                    new_lb.comment = 'BTA: perfect interpolated'
                    r.label = new_lb
            extended.append(r)
    return extended


def merge_ripeness_levels(records):
    """合并中间成熟度: RIPE→PERFECT, NEAR_OVERRIPE→OVERRIPE"""
    for r in records:
        if not r.is_labeled(): continue
        if r.label.ripeness_state == RipenessState.RIPE:
            r.label.ripeness_state = RipenessState.PERFECT
            if r.label.comment: r.label.comment += '; merged from RIPE'
            else: r.label.comment = 'merged from RIPE'
        elif r.label.ripeness_state == RipenessState.NEAR_OVERRIPE:
            r.label.ripeness_state = RipenessState.OVERRIPE
            if r.label.comment: r.label.comment += '; merged from NEAR_OVERRIPE'
            else: r.label.comment = 'merged from NEAR_OVERRIPE'
    return records


# ===== 应用 BTA =====
print('\n应用 BTA 双向时间假设...')
extended_set = apply_bta_to_fullset(full_set)
print(f'扩展后总计: {len(extended_set)} 条')
print(f'有标注: {sum(1 for r in extended_set if r.is_labeled())}')

# ===== 合并中间成熟度 =====
print('\n合并中间成熟度: RIPE→PERFECT, NEAR_OVERRIPE→OVERRIPE...')
extended_set = merge_ripeness_levels(extended_set)

# ===== 统计各类别（只算有标注的）=====
labeled_only = [r for r in extended_set if r.is_labeled()]
cnt = Counter()
for r in labeled_only:
    cnt[r.label.ripeness_state.name] += 1
print(f'标注分布: {dict(cnt)}')
bta_count = sum(1 for r in labeled_only if getattr(r.label, 'comment', '') and 'BTA' in r.label.comment)
print(f'其中 BTA 扩展的: {bta_count}')

# ===== 按水果ID随机划分（保证同一ID只在一个划分中）=====
id_to_records = defaultdict(list)
for r in labeled_only:
    id_to_records[r.id].append(r)
id_list = list(id_to_records.keys())
random.shuffle(id_list)

n_total = len(labeled_only)
n_val = int(n_total * 0.15)
n_test = int(n_total * 0.15)

val_ids, test_ids, train_ids = set(), set(), set()
val_count, test_count = 0, 0
for fid in id_list:
    n = len(id_to_records[fid])
    if val_count < n_val:
        val_ids.add(fid); val_count += n
    elif test_count < n_test:
        test_ids.add(fid); test_count += n
    else:
        train_ids.add(fid)

train_recs = []
for fid in train_ids: train_recs.extend(id_to_records[fid])
val_recs = []
for fid in val_ids: val_recs.extend(id_to_records[fid])
test_recs = []
for fid in test_ids: test_recs.extend(id_to_records[fid])

print(f'\n划分结果:')
print(f'  Train: {len(train_recs)} 条, {len(train_ids)} ID')
print(f'  Val:   {len(val_recs)} 条, {len(val_ids)} ID')
print(f'  Test:  {len(test_recs)} 条, {len(test_ids)} ID')

for name, recs in [('Train', train_recs), ('Val', val_recs), ('Test', test_recs)]:
    c = Counter()
    for r in recs: c[r.label.ripeness_state.name] += 1
    print(f'  {name}: {dict(c)}')

# ===== 加载数据（复用之前的函数）=====
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
            lm = {RipenessState.UNRIPE: 'unripe', RipenessState.PERFECT: 'perfect', RipenessState.OVERRIPE: 'overripe'}
            y.append(lm[r.label.ripeness_state])
            X.append(padded)
            fids.append(str(r.id))
        except Exception as e:
            continue
    return np.array(X), np.array(y), np.array(fids)

print('\n加载数据...')
X_tr, y_tr, _ = load_split(train_recs, data_path)
X_val, y_val, _ = load_split(val_recs, data_path)
X_te, y_te, te_fids = load_split(test_recs, data_path)

cnt = Counter(y_tr)
print(f'Train: {len(X_tr)} {dict(cnt)}')
print(f'Val:   {len(X_val)} {dict(Counter(y_val))}')
print(f'Test:  {len(X_te)} {dict(Counter(y_te))}')

# ===== 归一化 =====
mean = np.mean(X_tr, axis=(0,2,3), keepdims=True)
std = np.std(X_tr, axis=(0,2,3), keepdims=True)
std[std < 1e-6] = 1
for arr in [X_tr, X_val, X_te]:
    arr[:] = (arr - mean) / std
    np.clip(arr, -10, 10, out=arr)

# ===== 模型（同前）=====
class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        mid = max(channels // reduction, 4)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(nn.Linear(channels, mid), nn.ReLU(True), nn.Linear(mid, channels), nn.Sigmoid())
    def forward(self, x):
        w = self.fc(self.gap(x).flatten(1)).unsqueeze(-1).unsqueeze(-1); return x * w

class HSClassifier(nn.Module):
    def __init__(self, bands=224, num_classes=3, hidden=[64,128,256]):
        super().__init__(); k=3
        self.block1 = nn.Sequential(nn.Conv2d(bands, bands*k, 7, padding=3, groups=bands), nn.Conv2d(bands*k, hidden[0], 1), nn.ReLU(True), nn.AvgPool2d(2), nn.BatchNorm2d(hidden[0]), SEBlock(hidden[0]))
        self.block2 = nn.Sequential(nn.Conv2d(hidden[0], hidden[0]*k, 3, padding=1, groups=hidden[0]), nn.Conv2d(hidden[0]*k, hidden[1], 1), nn.ReLU(True), nn.AvgPool2d(2), nn.BatchNorm2d(hidden[1]), SEBlock(hidden[1]))
        self.block3 = nn.Sequential(nn.Conv2d(hidden[1], hidden[1]*k, 3, padding=1, groups=hidden[1]), nn.Conv2d(hidden[1]*k, hidden[2], 1), nn.ReLU(True), nn.BatchNorm2d(hidden[2]))
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(nn.Linear(hidden[2], hidden[2]), nn.BatchNorm1d(hidden[2]), nn.ReLU(True), nn.Dropout(0.3), nn.Linear(hidden[2], num_classes))
    def forward(self, x): return self.fc(self.gap(self.block3(self.block2(self.block1(x)))).flatten(1))

class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0): super().__init__(); self.gamma = gamma
    def forward(self, x, y):
        ce = nn.functional.cross_entropy(x, y, reduction='none')
        return (((1 - torch.exp(-ce))**self.gamma)*ce).mean()

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
    if random.random() > 0.5: x = torch.rot90(x, random.randint(1,3), [1,2])
    if random.random() > 0.5:
        h,w = x.shape[1], x.shape[2]; ch,cw = h//4, w//4
        y0 = random.randint(0, h-ch); x0 = random.randint(0, w-cw)
        x[:, y0:y0+ch, x0:x0+cw] = 0
    return x

def train_model(name, X_tr, y_tr, X_val, X_te, y_te, use_aug=True, use_focal=True):
    le = LabelEncoder()
    y_tr_t = le.fit_transform(y_tr)
    y_v_t = le.transform(y_val)
    y_te_t = le.transform(y_te)
    n_classes = len(le.classes_)

    tl = DataLoader(TensorDataset(torch.FloatTensor(X_tr), torch.LongTensor(y_tr_t)), BATCH_SIZE, shuffle=True, pin_memory=True)
    vl = DataLoader(TensorDataset(torch.FloatTensor(X_val), torch.LongTensor(y_v_t)), BATCH_SIZE, pin_memory=True)
    tel = DataLoader(TensorDataset(torch.FloatTensor(X_te), torch.LongTensor(y_te_t)), BATCH_SIZE, pin_memory=True)

    model = HSClassifier(num_classes=n_classes).to(device)
    opt = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(opt, step_size=10, gamma=0.5)
    criterion = FocalLoss(gamma=2.0) if use_focal else nn.CrossEntropyLoss()
    ema_obj = EMA(model, decay=0.9)

    best_acc = 0; best_ema = None; patience = 0
    for ep in range(EPOCHS):
        model.train()
        for x, y in tl:
            if use_aug: x = torch.stack([augment(xi) for xi in x])
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            opt.zero_grad(); criterion(model(x), y).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); ema_obj.update(model)
        scheduler.step()
        acc = evaluate(ema_obj.shadow, vl)
        if acc > best_acc: best_acc = acc; best_ema = copy.deepcopy(ema_obj.shadow.state_dict()); patience = 0
        else: patience += 1
        if patience >= PATIENCE: break

    ema_obj.shadow.load_state_dict(best_ema); ema_obj.shadow.eval()
    with torch.no_grad():
        probs = torch.cat([torch.softmax(ema_obj.shadow(x.to(device)),1).cpu() for x,_ in tel]).numpy()
    preds = probs.argmax(1)
    sa = accuracy_score(y_te_t, preds)
    f1 = f1_score(y_te_t, preds, average='weighted')
    print(f'  [{name:20s}] test={sa*100:.1f}%  f1={f1*100:.1f}%  val={best_acc*100:.1f}%  ep={ep+1}')
    return {'name': name, 'acc': round(float(sa),4), 'f1': round(float(f1),4), 'val': round(float(best_acc),4), 'n': len(X_tr)}

def evaluate(model, loader):
    model.eval(); correct, total = 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            correct += (model(x).argmax(1)==y).sum().item(); total += y.size(0)
    return correct/total

print(f'\n{"="*50}')
print('开始训练...')
print(f'{"="*50}')

results = []
results.append(train_model('BTA+Focal+Aug', X_tr, y_tr, X_val, X_te, y_te, True, True))
results.append(train_model('BTA+CE+Aug', X_tr, y_tr, X_val, X_te, y_te, True, False))
results.append(train_model('BTA+Focal无Aug', X_tr, y_tr, X_val, X_te, y_te, False, True))
results.append(train_model('BTA+CE无Aug', X_tr, y_tr, X_val, X_te, y_te, False, False))

print(f'\n{"="*50}')
print('BTA 修复版结果')
print(f'{"="*50}')
print(f'{"配置":25s} {"准确率":8s} {"F1":8s} {"训练集":6s}')
print('-'*50)
for r in sorted(results, key=lambda x: x['acc'], reverse=True):
    print(f'{r["name"]:25s} {r["acc"]*100:5.1f}%  {r["f1"]*100:5.1f}%  {r["n"]:4d}张')
print(f'{"="*50}')

# 保存
with open('bta_fixed_results.json', 'w') as f:
    json.dump(results, f, indent=2)
print(f'结果保存到 bta_fixed_results.json')
