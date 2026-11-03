"""
对比实验 - MLP (多层感知机)
使用展平的高光谱数据作为输入
"""
import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from collections import defaultdict
import json

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


class MLPClassifier(nn.Module):
    """多层感知机分类器"""
    def __init__(self, input_dim, hidden_dims=[512, 256, 128], num_classes=3, dropout=0.5):
        super().__init__()
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU(True))
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, num_classes))
        self.network = nn.Sequential(*layers)
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(self, x):
        # 输入 x: (batch, channels, height, width)
        # 展平: (batch, channels * height * width)
        x = x.view(x.size(0), -1)
        return self.network(x)


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0, 0, 0
    all_preds = []
    all_labels = []
    
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        pred = model(x)
        total_loss += criterion(pred, y).item()
        correct += (pred.argmax(1) == y).sum().item()
        total += y.size(0)
        all_preds.extend(pred.argmax(1).cpu().numpy())
        all_labels.extend(y.cpu().numpy())
    
    return total_loss / len(loader), correct / total, all_preds, all_labels


def fruit_level_acc(probs, fruit_ids, y):
    d = defaultdict(list)
    labels = {}
    for i, fid in enumerate(fruit_ids):
        d[fid].append(probs[i])
        labels[fid] = y[i]
    p = [np.stack(v).mean(0).argmax() for v in d.values()]
    return accuracy_score(list(labels.values()), p)


def main():
    print('='*60)
    print('对比实验: MLP (多层感知机)')
    print('='*60)
    
    # 加载数据
    data_path = os.path.join('..', 'ablation_study', 'ablation_data.npz')
    if not os.path.exists(data_path):
        print(f'错误: 数据文件不存在 {data_path}')
        return
    
    d = np.load(data_path, allow_pickle=True)
    X_train = torch.FloatTensor(d['X_train'])
    X_val = torch.FloatTensor(d['X_val'])
    X_test = torch.FloatTensor(d['X_test'])
    y_train, y_val, y_test = d['y_train'], d['y_val'], d['y_test']
    test_fruit_ids = d['test_fruit_ids']
    
    le = LabelEncoder()
    y_tr = le.fit_transform(y_train)
    y_v = le.transform(y_val)
    y_te = le.transform(y_test)
    
    print(f'类别映射: {dict(zip(le.classes_, range(len(le.classes_))))}')
    print(f'训练集: {X_train.shape}')
    print(f'验证集: {X_val.shape}')
    print(f'测试集: {X_test.shape}')
    
    # 计算输入维度
    input_dim = X_train.shape[1] * X_train.shape[2] * X_train.shape[3]
    print(f'输入维度: {input_dim}')
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'设备: {device}')
    print('='*60)
    
    # 数据加载器
    train_loader = DataLoader(
        TensorDataset(X_train, torch.LongTensor(y_tr)),
        batch_size=BATCH_SIZE, shuffle=True
    )
    val_loader = DataLoader(
        TensorDataset(X_val, torch.LongTensor(y_v)),
        batch_size=BATCH_SIZE, shuffle=False
    )
    test_loader = DataLoader(
        TensorDataset(X_test, torch.LongTensor(y_te)),
        batch_size=BATCH_SIZE, shuffle=False
    )
    
    # 模型
    model = MLPClassifier(
        input_dim=input_dim,
        hidden_dims=[512, 256, 128],
        num_classes=len(le.classes_),
        dropout=0.5
    ).to(device)
    
    print(model)
    print('='*60)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=BASE_LR, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=10, factor=0.5)
    
    # 训练
    best_val_loss = float('inf')
    best_state = None
    patience_counter = 0
    
    print('开始训练...')
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        val_loss, val_acc, _, _ = evaluate(model, val_loader, criterion, device)
        scheduler.step(val_loss)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
        
        if (epoch + 1) % 10 == 0:
            print(f'Epoch {epoch+1:3d} | Train Loss: {train_loss/len(train_loader):.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}')
        
        if patience_counter >= PATIENCE:
            print(f'早停于 Epoch {epoch+1}')
            break
    
    # 测试
    model.load_state_dict(best_state)
    test_loss, test_acc, preds, labels = evaluate(model, test_loader, criterion, device)
    
    # 水果级别准确率
    model.eval()
    all_probs = []
    with torch.no_grad():
        for x, _ in test_loader:
            x = x.to(device)
            probs = torch.softmax(model(x), dim=1).cpu().numpy()
            all_probs.append(probs)
    probs = np.concatenate(all_probs)
    fruit_acc = fruit_level_acc(probs, test_fruit_ids, y_te)
    
    print('\n' + '='*60)
    print('测试结果')
    print('='*60)
    print(f'Test Loss: {test_loss:.4f}')
    print(f'Sample-Level Accuracy: {test_acc:.4f}')
    print(f'Fruit-Level Accuracy:  {fruit_acc:.4f}')
    
    print('\n分类报告:')
    print(classification_report(labels, preds, target_names=le.classes_))
    
    print('混淆矩阵:')
    cm = confusion_matrix(labels, preds)
    print(cm)
    
    # 保存结果
    results = {
        'model': 'MLP',
        'input_dim': input_dim,
        'hidden_dims': [512, 256, 128],
        'sample_acc': float(test_acc),
        'fruit_acc': float(fruit_acc),
        'class_report': classification_report(labels, preds, target_names=le.classes_, output_dict=True),
        'confusion_matrix': cm.tolist()
    }
    
    with open('results_mlp.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print('\n结果已保存到 results_mlp.json')
    print('='*60)


if __name__ == '__main__':
    main()
