"""多种子对比 SE vs CA"""
import os, json, copy, random
import numpy as np
import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from collections import defaultdict

EPOCHS = 80; PATIENCE = 15; BATCH_SIZE = 32; BASE_LR = 1e-3
WEIGHT_DECAY = 1e-4; GRAD_CLIP = 1.0

class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        mid = max(channels // reduction, 4)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(nn.Linear(channels, mid), nn.ReLU(True), nn.Linear(mid, channels), nn.Sigmoid())
    def forward(self, x):
        w = self.fc(self.gap(x).flatten(1)).unsqueeze(-1).unsqueeze(-1)
        return x * w

class CABlock(nn.Module):
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
        b, c, h, w = x.shape
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)
        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y); y = self.bn1(y); y = self.act(y)
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)
        a_h = self.sigmoid(self.conv_h(x_h))
        a_w = self.sigmoid(self.conv_w(x_w))
        return x * a_h * a_w

def make_net(bands, num_classes, attn_type='none'):
    hidden = [25, 30, 50]; k = 3
    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            a1 = SEBlock(hidden[0]) if attn_type=='se' else CABlock(hidden[0]) if attn_type=='ca' else nn.Identity()
            a2 = SEBlock(hidden[1]) if attn_type=='se' else CABlock(hidden[1]) if attn_type=='ca' else nn.Identity()
            self.block1 = nn.Sequential(nn.Conv2d(bands, bands*k, 7, padding=3, groups=bands), nn.Conv2d(bands*k, hidden[0], 1), nn.ReLU(True), nn.AvgPool2d(2), nn.BatchNorm2d(hidden[0]), a1)
            self.block2 = nn.Sequential(nn.Conv2d(hidden[0], hidden[0]*k, 3, padding=1, groups=hidden[0]), nn.Conv2d(hidden[0]*k, hidden[1], 1), nn.ReLU(True), nn.AvgPool2d(2), nn.BatchNorm2d(hidden[1]), a2)
            self.block3 = nn.Sequential(nn.Conv2d(hidden[1], hidden[1]*k, 3, padding=1, groups=hidden[1]), nn.Conv2d(hidden[1]*k, hidden[2], 1), nn.ReLU(True), nn.BatchNorm2d(hidden[2]))
            self.gap = nn.AdaptiveAvgPool2d(1)
            self.fc = nn.Sequential(nn.Linear(hidden[2], hidden[2]), nn.BatchNorm1d(hidden[2]), nn.ReLU(True), nn.Dropout(0.3), nn.Linear(hidden[2], num_classes))
        def forward(self, x):
            return self.fc(self.gap(self.block3(self.block2(self.block1(x)))).flatten(1))
    return Net()

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

def train_one(seed, attn_type):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    data = np.load('ablation_data.npz', allow_pickle=True)
    X_train = torch.FloatTensor(np.array(data['X_train_vis'], dtype=np.float32, copy=True))
    X_val = torch.FloatTensor(np.array(data['X_val_vis'], dtype=np.float32, copy=True))
    X_test = torch.FloatTensor(np.array(data['X_test_vis'], dtype=np.float32, copy=True))
    y_train, y_val, y_test = data['y_train_vis'], data['y_val_vis'], data['y_test_vis']
    test_fruit_ids = data['test_fruit_ids']
    del data

    le = LabelEncoder()
    y_tr = le.fit_transform(y_train); y_v = le.transform(y_val); y_te = le.transform(y_test)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    bands = X_train.shape[1]; n_classes = len(le.classes_)

    tl = DataLoader(TensorDataset(X_train, torch.LongTensor(y_tr)), BATCH_SIZE, shuffle=True)
    vl = DataLoader(TensorDataset(X_val, torch.LongTensor(y_v)), BATCH_SIZE)
    tel = DataLoader(TensorDataset(X_test, torch.LongTensor(y_te)), BATCH_SIZE)

    model = make_net(bands, n_classes, attn_type=attn_type).to(device)
    opt = optim.AdamW(model.parameters(), lr=BASE_LR, weight_decay=WEIGHT_DECAY)
    criterion = nn.CrossEntropyLoss()
    ema_obj = EMA(model, decay=0.9)

    best_acc = 0; best_sd = None; best_ema_sd = None; patience = 0
    for ep in range(EPOCHS):
        model.train()
        for x, y in tl:
            x, y = x.to(device), y.to(device)
            opt.zero_grad(); criterion(model(x), y).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            opt.step(); ema_obj.update(model)
        acc = evaluate(ema_obj.shadow, vl, device)
        if acc > best_acc:
            best_acc = acc; best_sd = copy.deepcopy(model.state_dict())
            best_ema_sd = copy.deepcopy(ema_obj.shadow.state_dict()); patience = 0
        else: patience += 1
        if patience >= PATIENCE:
            break

    ema_obj.shadow.load_state_dict(best_ema_sd)
    inf_model = ema_obj.shadow; inf_model.eval()

    with torch.no_grad():
        probs = torch.cat([torch.softmax(inf_model(x.to(device)), 1).cpu() for x, _ in tel]).numpy()
    sa = accuracy_score(y_te, probs.argmax(1))
    fa = fruit_level_acc(probs, test_fruit_ids, y_te)
    return sa, fa, best_acc, ep+1

print("=" * 55)
print("SE vs CA 多种子对比 (80 epochs, EMA eval)")
print("=" * 55)
header = f"{'Seed':>6s}  {'SE Acc':>8s} {'SE Fruit':>9s}  {'CA Acc':>8s} {'CA Fruit':>9s}  {'Winner':>8s}"
print(header)
print("-" * len(header))

se_scores = []
ca_scores = []

for seed in [24, 42, 123, 7, 999, 111]:
    sa_se, fa_se, va_se, ep_se = train_one(seed, 'se')
    sa_ca, fa_ca, va_ca, ep_ca = train_one(seed, 'ca')
    se_scores.append(sa_se)
    ca_scores.append(sa_ca)
    winner = "CA" if sa_ca > sa_se else "SE" if sa_se > sa_ca else "TIE"
    mark = " <<" if winner == "CA" else ""
    print(f"{seed:6d}  {sa_se*100:6.1f}%  {fa_se*100:7.1f}%  {sa_ca*100:6.1f}%  {fa_ca*100:7.1f}%  {winner:>8s}{mark}")

print("-" * len(header))
print(f"{'MEAN':>6s}  {np.mean(se_scores)*100:6.1f}%  {'':9s}  {np.mean(ca_scores)*100:6.1f}%")
print(f"{'STD':>6s}  {np.std(se_scores)*100:6.1f}%  {'':9s}  {np.std(ca_scores)*100:6.1f}%")
print(f"\nSE wins: {sum(1 for s,c in zip(se_scores, ca_scores) if s>c)}/{len(se_scores)}")
print(f"CA wins: {sum(1 for s,c in zip(se_scores, ca_scores) if c>s)}/{len(ca_scores)}")
