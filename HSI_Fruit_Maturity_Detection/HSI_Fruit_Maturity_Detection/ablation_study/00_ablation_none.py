"""
消融实验 - 基线模型 (无任何优化组件)
- 无 SE Block
- 无 EMA
- 无 TTA
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


class IdentityBlock(nn.Module):
    def forward(self, x):
        return x


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
            IdentityBlock()
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(hidden[0], hidden[0]*k, 3, padding=1, groups=hidden[0]),
            nn.Conv2d(hidden[0]*k, hidden[1], 1),
            nn.ReLU(True),
            nn.AvgPool2d(2),
            nn.BatchNorm2d(hidden[1]),
            IdentityBlock()
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

best_val_loss = float('inf')
best_sd = None
patience_counter = 0

print('00_ablation_none: 基线模型 (无SE, 无EMA, 无TTA)')
for epoch in range(EPOCHS):
    model.train()
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        loss = train_criterion(model(x), y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRAD_CLIP)
        optimizer.step()

    val_loss, val_acc = evaluate(model, val_loader, val_criterion, device)

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_sd = copy.deepcopy(model.state_dict())
        patience_counter = 0
    else:
        patience_counter += 1

    if (epoch + 1) % 10 == 0:
        print(f'Epoch {epoch+1:3d} | val_acc={val_acc:.4f}')

    if patience_counter >= PATIENCE:
        print(f'Early stopping at epoch {epoch+1}')
        break

inf_model = ClassifierNetwork(X_train.shape[1], len(le.classes_)).to(device)
inf_model.load_state_dict(best_sd)
inf_model.eval()

preds_list = []
with torch.no_grad():
    for x, _ in test_loader:
        x = x.to(device)
        preds_list.append(torch.softmax(inf_model(x), dim=1).cpu())
probs = torch.cat(preds_list).numpy()
preds = probs.argmax(1)

sample_acc = accuracy_score(y_te, preds)
fruit_acc = fruit_level_acc(probs, test_fruit_ids, y_te)

print('\n' + '='*55)
print('00_ablation_none Results:')
print(f'Sample-Level Accuracy: {sample_acc:.4f}')
print(f'Fruit-Level Accuracy:  {fruit_acc:.4f}')
print('='*55)

# 保存结果
import json
results = {
    'experiment': '00_ablation_none',
    'components': {'SE': False, 'EMA': False, 'TTA': False},
    'sample_acc': float(sample_acc),
    'fruit_acc': float(fruit_acc)
}
with open('results_00_ablation_none.json', 'w') as f:
    json.dump(results, f, indent=2)
