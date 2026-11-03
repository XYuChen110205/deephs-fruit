"""
消融实验 - 完整模型 (SE + EMA + TTA)
- 有 SE Block
- 有 EMA
- 有 TTA
"""
import os
import random
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from collections import defaultdict

SEED = 24
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

BASE_LR = 1e-3
WEIGHT_DECAY = 1e-4
BATCH_SIZE = 32
EPOCHS = 150
PATIENCE = 30
GRAD_CLIP = 1.0
EMA_WARMUP_EPOCHS = 15
EMA_INIT_DECAY = 0.9
EMA_FINAL_DECAY = 0.999
TTA_ITER = 5


class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        mid = max(channels // reduction, 4)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, mid),
            nn.ReLU(True),
            nn.Linear(mid, channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        w = self.fc(self.gap(x).flatten(1)).unsqueeze(-1).unsqueeze(-1)
        return x * w


class ClassifierNetwork(nn.Module):
    def __init__(self, bands, num_classes=3):
        super().__init__()
        hidden = [25, 30, 50]
        k = 3
        self.block1 = nn.Sequential(
            nn.Conv2d(bands, bands*k, 7, padding=3, groups=bands),
            nn.Conv2d(bands*k, hidden[0], 1),
            nn.ReLU(True),
            nn.AvgPool2d(2),
            nn.BatchNorm2d(hidden[0]),
            SEBlock(hidden[0])
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(hidden[0], hidden[0]*k, 3, padding=1, groups=hidden[0]),
            nn.Conv2d(hidden[0]*k, hidden[1], 1),
            nn.ReLU(True),
            nn.AvgPool2d(2),
            nn.BatchNorm2d(hidden[1]),
            SEBlock(hidden[1])
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(hidden[1], hidden[1]*k, 3, padding=1, groups=hidden[1]),
            nn.Conv2d(hidden[1]*k, hidden[2], 1),
            nn.ReLU(True),
            nn.BatchNorm2d(hidden[2])
        )
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(hidden[2], hidden[2]),
            nn.BatchNorm1d(hidden[2]),
            nn.ReLU(True),
            nn.Dropout(0.3),
            nn.Linear(hidden[2], num_classes)
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.gap(x)
        x = x.flatten(1)
        x = self.fc(x)
        return x


class EMA:
    def __init__(self, model, init_decay=0.9):
        self.decay = init_decay
        self.shadow = copy.deepcopy(model).eval()
        for p in self.shadow.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        for s_p, m_p in zip(self.shadow.parameters(), model.parameters()):
            s_p.data.mul_(self.decay).add_(m_p.data, alpha=1 - self.decay)
        for s_b, m_b in zip(self.shadow.buffers(), model.buffers()):
            s_b.copy_(m_b)

    def set_decay(self, d):
        self.decay = d

    def eval_model(self):
        return self.shadow


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        pred = model(x)
        total_loss += criterion(pred, y).item()
        correct += (pred.argmax(1) == y).sum().item()
        total += y.size(0)
    return total_loss / len(loader), correct / total


def fruit_level_acc(probs, fruit_ids, y):
    d = defaultdict(list)
    labels = {}
    for i, fid in enumerate(fruit_ids):
        d[fid].append(probs[i])
        labels[fid] = y[i]
    p = [np.stack(v).mean(0).argmax() for v in d.values()]
    return accuracy_score(list(labels.values()), p)


d = np.load('ablation_data.npz', allow_pickle=True)
X_train = torch.FloatTensor(d['X_train_vis'])
X_val = torch.FloatTensor(d['X_val_vis'])
X_test = torch.FloatTensor(d['X_test_vis'])
y_train, y_val, y_test = d['y_train_vis'], d['y_val_vis'], d['y_test_vis']
test_fruit_ids = d['test_fruit_ids']

le = LabelEncoder()
y_tr = le.fit_transform(y_train)
y_v = le.transform(y_val)
y_te = le.transform(y_test)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'Device: {device}')

train_loader = DataLoader(TensorDataset(X_train, torch.LongTensor(y_tr)), batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(TensorDataset(X_val, torch.LongTensor(y_v)), batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(TensorDataset(X_test, torch.LongTensor(y_te)), batch_size=BATCH_SIZE, shuffle=False)

model = ClassifierNetwork(bands=X_train.shape[1], num_classes=len(le.classes_)).to(device)
train_criterion = nn.CrossEntropyLoss()
val_criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=BASE_LR, weight_decay=WEIGHT_DECAY)
ema = EMA(model, init_decay=EMA_INIT_DECAY)

best_val_loss = float('inf')
best_sd = None
best_ema_sd = None
patience_counter = 0

print('03_ablation_full: 完整模型 (SE + EMA + TTA)')
for epoch in range(EPOCHS):
    model.train()
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        loss = train_criterion(model(x), y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRAD_CLIP)
        optimizer.step()
        ema.update(model)

    if epoch < EMA_WARMUP_EPOCHS:
        cur_ema_decay = EMA_INIT_DECAY
    else:
        ramp = min((epoch - EMA_WARMUP_EPOCHS) / max(EPOCHS - EMA_WARMUP_EPOCHS, 1), 1.0)
        cur_ema_decay = EMA_INIT_DECAY + ramp * (EMA_FINAL_DECAY - EMA_INIT_DECAY)
    ema.set_decay(cur_ema_decay)

    raw_val_loss, raw_val_acc = evaluate(model, val_loader, val_criterion, device)
    ema_val_loss, ema_val_acc = evaluate(ema.eval_model(), val_loader, val_criterion, device)

    if epoch < EMA_WARMUP_EPOCHS:
        tracked_val_loss = raw_val_loss
    else:
        tracked_val_loss = ema_val_loss

    if tracked_val_loss < best_val_loss:
        best_val_loss = tracked_val_loss
        best_sd = copy.deepcopy(model.state_dict())
        best_ema_sd = copy.deepcopy(ema.shadow.state_dict())
        patience_counter = 0
    else:
        patience_counter += 1

    if (epoch + 1) % 10 == 0:
        print(f'Epoch {epoch+1:3d} | raw_acc={raw_val_acc:.4f} ema_acc={ema_val_acc:.4f}')

    if patience_counter >= PATIENCE:
        print(f'Early stopping at epoch {epoch+1}')
        break

if best_ema_sd is not None:
    ema.shadow.load_state_dict(best_ema_sd)
inf_model = ema.eval_model()
inf_model.eval()

all_probs = []
for _ in range(TTA_ITER):
    preds_list = []
    with torch.no_grad():
        for x, _ in test_loader:
            x = x.to(device)
            x = x.clone()
            for i in range(len(x)):
                if torch.rand(1) > 0.5:
                    x[i] = torch.flip(x[i], [1])
                if torch.rand(1) > 0.5:
                    x[i] = torch.flip(x[i], [2])
                if torch.rand(1) > 0.5:
                    x[i] = torch.rot90(x[i], 1, [1, 2])
            preds_list.append(torch.softmax(inf_model(x), dim=1).cpu())
    all_probs.append(torch.cat(preds_list))
probs = torch.stack(all_probs).mean(0).numpy()
preds = probs.argmax(1)

sample_acc = accuracy_score(y_te, preds)
fruit_acc = fruit_level_acc(probs, test_fruit_ids, y_te)

print('\n' + '='*55)
print('03_ablation_full Results:')
print(f'Sample-Level Accuracy: {sample_acc:.4f}')
print(f'Fruit-Level Accuracy:  {fruit_acc:.4f}')
print('='*55)

# 保存结果
import json
results = {
    'experiment': '03_ablation_full',
    'components': {'SE': True, 'EMA': True, 'TTA': True},
    'sample_acc': float(sample_acc),
    'fruit_acc': float(fruit_acc)
}
with open('results_03_ablation_full.json', 'w') as f:
    json.dump(results, f, indent=2)
