"""
消融实验扩展版 — 对比 SE vs CA (Coordinate Attention)
注意：本文件不破坏任何旧文件，结果输出到单独文件
"""
import os, json, copy, random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from collections import defaultdict

SEED = 24
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

EPOCHS = 30; PATIENCE = 10; BATCH_SIZE = 32; BASE_LR = 1e-3
WEIGHT_DECAY = 1e-4; GRAD_CLIP = 1.0


# ===================== SE Block =====================
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


# ===================== CA Block (Coordinate Attention) =====================
class CABlock(nn.Module):
    """Coordinate Attention for Efficient Mobile Network Design (CVPR 2021)"""
    def __init__(self, channels, reduction=16):
        super().__init__()
        mid = max(channels // reduction, 4)
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        self.conv1 = nn.Conv2d(channels, mid, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(mid)
        self.act = nn.ReLU(True)
        self.conv_h = nn.Conv2d(mid, channels, kernel_size=1, bias=False)
        self.conv_w = nn.Conv2d(mid, channels, kernel_size=1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        identity = x
        b, c, h, w = x.shape
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)
        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)
        a_h = self.sigmoid(self.conv_h(x_h))
        a_w = self.sigmoid(self.conv_w(x_w))
        return identity * a_h * a_w


# ===================== 网络构建 =====================
def make_net(bands, num_classes, attn_type='none'):
    """attn_type: 'none' / 'se' / 'ca' """
    hidden = [25, 30, 50]; k = 3
    def attn_fn(ch):
        if attn_type == 'se': return SEBlock(ch)
        elif attn_type == 'ca': return CABlock(ch)
        else: return nn.Identity()

    class HSClassifier(nn.Module):
        def __init__(self):
            super().__init__()
            self.block1 = nn.Sequential(
                nn.Conv2d(bands, bands*k, 7, padding=3, groups=bands),
                nn.Conv2d(bands*k, hidden[0], 1), nn.ReLU(True), nn.AvgPool2d(2),
                nn.BatchNorm2d(hidden[0]), attn_fn(hidden[0]))
            self.block2 = nn.Sequential(
                nn.Conv2d(hidden[0], hidden[0]*k, 3, padding=1, groups=hidden[0]),
                nn.Conv2d(hidden[0]*k, hidden[1], 1), nn.ReLU(True), nn.AvgPool2d(2),
                nn.BatchNorm2d(hidden[1]), attn_fn(hidden[1]))
            self.block3 = nn.Sequential(
                nn.Conv2d(hidden[1], hidden[1]*k, 3, padding=1, groups=hidden[1]),
                nn.Conv2d(hidden[1]*k, hidden[2], 1), nn.ReLU(True), nn.BatchNorm2d(hidden[2]))
            self.gap = nn.AdaptiveAvgPool2d(1)
            self.fc = nn.Sequential(
                nn.Linear(hidden[2], hidden[2]), nn.BatchNorm1d(hidden[2]),
                nn.ReLU(True), nn.Dropout(0.3), nn.Linear(hidden[2], num_classes))
        def forward(self, x):
            x = self.block1(x); x = self.block2(x); x = self.block3(x)
            x = self.gap(x).flatten(1); x = self.fc(x)
            return x
    return HSClassifier()


class EMA:
    def __init__(self, model, decay=0.9):
        self.decay = decay
        self.shadow = copy.deepcopy(model).eval()
        for p in self.shadow.parameters(): p.requires_grad_(False)
    def update(self, model):
        for s_p, m_p in zip(self.shadow.parameters(), model.parameters()):
            s_p.data.mul_(self.decay).add_(m_p.data, alpha=1-self.decay)
        for s_b, m_b in zip(self.shadow.buffers(), model.buffers()):
            s_b.copy_(m_b)

def evaluate(model, loader, device):
    model.eval(); correct, total = 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            correct += (model(x).argmax(1) == y).sum().item()
            total += y.size(0)
    return correct / total

def fruit_level_acc(probs, fruit_ids, y):
    d = defaultdict(list); labels = {}
    for i, fid in enumerate(fruit_ids):
        d[fid].append(probs[i]); labels[fid] = y[i]
    p = [np.stack(v).mean(0).argmax() for v in d.values()]
    return accuracy_score(list(labels.values()), p)


# ===================== 加载数据（内存优化版）=====================
data = np.load('ablation_data.npz', allow_pickle=True)
# 每次只保留一个数组，减少内存占用
X_train = np.array(data['X_train_vis'], dtype=np.float32, copy=True)
X_val = np.array(data['X_val_vis'], dtype=np.float32, copy=True)
X_test = np.array(data['X_test_vis'], dtype=np.float32, copy=True)
y_train = data['y_train_vis'].copy()
y_val = data['y_val_vis'].copy()
y_test = data['y_test_vis'].copy()
test_fruit_ids = data['test_fruit_ids'].copy()
del data

X_train = torch.FloatTensor(X_train)
X_val = torch.FloatTensor(X_val)
X_test = torch.FloatTensor(X_test)

le = LabelEncoder()
y_tr = le.fit_transform(y_train)
y_v, y_te = le.transform(y_val), le.transform(y_test)
n_classes = len(le.classes_)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
bands = X_train.shape[1]
print(f'Device: {device}')
print(f'Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}')

train_loader = DataLoader(TensorDataset(X_train, torch.LongTensor(y_tr)), BATCH_SIZE, shuffle=True)
val_loader = DataLoader(TensorDataset(X_val, torch.LongTensor(y_v)), BATCH_SIZE)
test_loader = DataLoader(TensorDataset(X_test, torch.LongTensor(y_te)), BATCH_SIZE)


# ===================== 训练 =====================
def train_model(attn_type, use_ema=True, use_tta=True):
    """训练一个配置，返回准确率"""
    print(f'\n=== attn={attn_type}, ema={use_ema}, tta={use_tta} ===')
    model = make_net(bands, n_classes, attn_type=attn_type).to(device)
    opt = optim.AdamW(model.parameters(), lr=BASE_LR, weight_decay=WEIGHT_DECAY)
    criterion = nn.CrossEntropyLoss()
    ema_obj = EMA(model, decay=0.9) if use_ema else None

    best_acc = 0; best_sd = None; best_ema_sd = None; patience = 0
    for ep in range(EPOCHS):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad(); criterion(model(x), y).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            opt.step()
            if ema_obj: ema_obj.update(model)

        if ema_obj: acc = evaluate(ema_obj.shadow, val_loader, device)
        else: acc = evaluate(model, val_loader, device)

        if acc > best_acc:
            best_acc = acc; best_sd = copy.deepcopy(model.state_dict())
            if ema_obj: best_ema_sd = copy.deepcopy(ema_obj.shadow.state_dict())
            patience = 0
        else: patience += 1
        if patience >= PATIENCE:
            print(f'  Early stop at epoch {ep+1}')
            break

    if ema_obj and best_ema_sd is not None:
        ema_obj.shadow.load_state_dict(best_ema_sd); inf_model = ema_obj.shadow
    else:
        model.load_state_dict(best_sd); inf_model = model
    inf_model.eval()

    if use_tta:
        all_probs = []
        for _ in range(5):
            preds_list = []
            with torch.no_grad():
                for x, _ in test_loader:
                    x = x.to(device); xi = x.clone()
                    for i in range(len(xi)):
                        if torch.rand(1) > 0.5: xi[i] = torch.flip(xi[i], [1])
                        if torch.rand(1) > 0.5: xi[i] = torch.flip(xi[i], [2])
                        if torch.rand(1) > 0.5: xi[i] = torch.rot90(xi[i], 1, [1, 2])
                    preds_list.append(torch.softmax(inf_model(xi), 1).cpu())
            all_probs.append(torch.cat(preds_list))
        probs = torch.stack(all_probs).mean(0).numpy()
    else:
        with torch.no_grad():
            probs = torch.cat([torch.softmax(inf_model(x.to(device)), 1).cpu() for x, _ in test_loader]).numpy()

    sa = accuracy_score(y_te, probs.argmax(1))
    fa = fruit_level_acc(probs, test_fruit_ids, y_te)
    val_acc = best_acc
    print(f'  Sample Acc: {sa:.4f}, Fruit Acc: {fa:.4f}, Val Acc: {val_acc:.4f}')
    return sa, fa, val_acc


# ===================== 运行所有配置 =====================
configs = [
    ('none', False, False, 'Baseline'),
    ('se',   False, False, 'SE Only'),
    ('ca',   False, False, 'CA Only'),
    ('se',   True,  True,  'SE+EMA+TTA'),
    ('ca',   True,  True,  'CA+EMA+TTA'),
]

results = []
for attn, use_ema, use_tta, name in configs:
    sa, fa, va = train_model(attn, use_ema, use_tta)
    results.append({
        'name': name, 'attention': attn,
        'ema': use_ema, 'tta': use_tta,
        'sample_acc': round(float(sa), 4),
        'fruit_acc': round(float(fa), 4),
        'val_acc': round(float(va), 4),
    })

# ===================== 输出结果 =====================
print('\n\n' + '='*70)
print('注意力机制对比实验结果')
print('='*70)
print(f'{"配置":25s} {"样本准确率":12s} {"水果准确率":12s} {"验证准确率":12s}')
print('-'*70)
for r in results:
    print(f'{r["name"]:25s} {r["sample_acc"]*100:8.1f}%    {r["fruit_acc"]*100:8.1f}%    {r["val_acc"]*100:8.1f}%')

# 保存结果
out_path = 'attention_comparison_results.json'
output = {
    'dataset': {
        'total_samples': int(X_train.shape[0]),
        'n_bands': bands,
        'n_classes': n_classes,
    },
    'results': results,
}
with open(out_path, 'w') as f:
    json.dump(output, f, indent=2)
print(f'\n结果已保存到 {out_path}')
