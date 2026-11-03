"""
复现你队友的配置 — 原始论文标准训练流程
直接用我们已验证的代码，参数完全按论文来
"""
import os, sys, copy, random, json, time
import numpy as np
import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from collections import Counter

SEED = 23312323
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
BATCH_SIZE = 64; EPOCHS = 300; PATIENCE = 40
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'设备: {device}')

sys.path.insert(0, r'e:\DeepHS-Fruit\deephs_fruit')
from core.datasets.hyperspectral_dataset import get_records, extend_unripe_recordings
from core.name_convention import *
import spectral

data_path = r'e:\DeepHS-Fruit\deephs_fruit\data'

# ===== 论文原始方式加载 =====
print('\n=== 按原始论文方式加载数据 ===')
# 方式1: 原始随机划分，无时间假设（132张）
train1, val1, test1 = get_records(Fruit.AVOCADO, CameraType.VIS, ClassificationType.RIPENESS,
    use_inter_ripeness_levels=True, extend_by_time_assumption=False,
    random_split=True, seed=SEED)

# 方式2: 原始随机划分 + 论文时间假设（只推unripe，446张）
# get_records 直接传 extend_by_time_assumption=True 即可
train2, val2, test2 = get_records(Fruit.AVOCADO, CameraType.VIS, ClassificationType.RIPENESS,
    use_inter_ripeness_levels=True, extend_by_time_assumption=True,
    random_split=True, seed=SEED)

for name, tr, va, te in [('原始划分(无时间假设)', train1, val1, test1), 
                           ('原始划分+论文时间假设', train2, val2, test2)]:
    for sn, recs in [('Train', tr), ('Val', va), ('Test', te)]:
        cnt = Counter(r.label.ripeness_state.name for r in recs if r.is_labeled())
        print(f'{name:30s} {sn}: {len(recs):3d} {dict(cnt)}')

def load_split(records, data_path):
    X, y, fids = [], [], []
    for r in records:
        if not r.is_labeled(): continue
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
            padding = np.zeros((224, 64, 64), dtype=np.float32)
            padding[:, 1:-1, 1:-1] = t.numpy()
            y.append(r.label.ripeness_state.name.lower())
            X.append(padding)
            fids.append(str(r.id))
        except: continue
    return np.array(X), np.array(y), np.array(fids)

print('\n=== 加载数据 ===')
print('加载原始划分(132张)...')
X1_tr, y1_tr, _ = load_split(train1, data_path)
X1_te, y1_te, t1_fids = load_split(test1, data_path)
X1_va, y1_va, _ = load_split(val1, data_path)

print('加载论文时间假设扩展...')
X2_tr, y2_tr, _ = load_split(train2, data_path)
# 测试集和验证集用同一份（论文方式）
X2_te, y2_te = X1_te.copy(), y1_te.copy()
X2_va, y2_va = X1_va.copy(), y1_va.copy()

for name, X, y in [('原始Train', X1_tr, y1_tr), ('原始Test', X1_te, y1_te),
                     ('论文Train', X2_tr, y2_tr), ('论文Test', X2_te, y2_te)]:
    cnt = Counter(y)
    print(f'  {name:15s}: {len(X):4d} {dict(cnt)}')

# ===== 归一化 =====
def normalize(X_tr, X_va, X_te):
    m = np.mean(X_tr, axis=(0,2,3), keepdims=True)
    s = np.std(X_tr, axis=(0,2,3), keepdims=True)
    s[s<1e-6]=1
    for arr in [X_tr, X_va, X_te]:
        arr[:] = (arr - m) / s; np.clip(arr, -10, 10, out=arr)

normalize(X1_tr, X1_va, X1_te)
normalize(X2_tr, X2_va, X2_te)

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
    def __init__(self, b=224, nc=3, h=[64,128,256]):
        super().__init__(); k=3
        self.b1=nn.Sequential(nn.Conv2d(b,b*k,7,padding=3,groups=b),nn.Conv2d(b*k,h[0],1),nn.ReLU(True),nn.AvgPool2d(2),nn.BatchNorm2d(h[0]),SEBlock(h[0]))
        self.b2=nn.Sequential(nn.Conv2d(h[0],h[0]*k,3,padding=1,groups=h[0]),nn.Conv2d(h[0]*k,h[1],1),nn.ReLU(True),nn.AvgPool2d(2),nn.BatchNorm2d(h[1]),SEBlock(h[1]))
        self.b3=nn.Sequential(nn.Conv2d(h[1],h[1]*k,3,padding=1,groups=h[1]),nn.Conv2d(h[1]*k,h[2],1),nn.ReLU(True),nn.BatchNorm2d(h[2]))
        self.gap=nn.AdaptiveAvgPool2d(1); self.fc=nn.Sequential(nn.Linear(h[2],h[2]),nn.BatchNorm1d(h[2]),nn.ReLU(True),nn.Dropout(0.3),nn.Linear(h[2],nc))
    def forward(self,x): return self.fc(self.gap(self.b3(self.b2(self.b1(x)))).flatten(1))

class FocalLoss(nn.Module):
    def __init__(self,g=2.0): super().__init__();self.g=g
    def forward(self,x,y):ce=nn.functional.cross_entropy(x,y,reduction='none');return(((1-torch.exp(-ce))**self.g)*ce).mean()

class EMA:
    def __init__(self,m,d=0.9):self.d=d;self.s=copy.deepcopy(m).eval();[p.requires_grad_(False)for p in self.s.parameters()]
    def update(self,m):[sp.data.mul_(self.d).add_(mp.data,alpha=1-self.d)for sp,mp in zip(self.s.parameters(),m.parameters())];[sb.copy_(mb)for sb,mb in zip(self.s.buffers(),m.buffers())]

def augment(x):
    if random.random()>0.5:x=torch.flip(x,[1])
    if random.random()>0.5:x=torch.rot90(x,random.randint(1,3),[1,2])
    if random.random()>0.5:h,w=x.shape[1],x.shape[2];ch,cw=h//4,w//4;y0=random.randint(0,h-ch);x0=random.randint(0,w-cw);x[:,y0:y0+ch,x0:x0+cw]=0
    return x

def train(name, X_tr, y_tr, X_va, X_te, y_te):
    le = LabelEncoder()
    y_tr_t=le.fit_transform(y_tr); y_v_t=le.transform(y_va); y_te_t=le.transform(y_te)
    nc=len(le.classes_)
    tl=DataLoader(TensorDataset(torch.FloatTensor(X_tr),torch.LongTensor(y_tr_t)),BATCH_SIZE,shuffle=True,pin_memory=True)
    vl=DataLoader(TensorDataset(torch.FloatTensor(X_va),torch.LongTensor(y_v_t)),BATCH_SIZE,pin_memory=True)
    tel=DataLoader(TensorDataset(torch.FloatTensor(X_te),torch.LongTensor(y_te_t)),BATCH_SIZE,pin_memory=True)
    m=HSClassifier(nc=nc).to(device)
    opt=optim.SGD(m.parameters(),lr=0.01,momentum=0.9,weight_decay=1e-4)
    sch=optim.lr_scheduler.StepLR(opt,10,0.5)
    cri=FocalLoss(2.0); ema=EMA(m,0.9)
    ba=0;be=None;pa=0
    for ep in range(EPOCHS):
        m.train()
        for x,y in tl:
            if random.random()>0.5:x=torch.stack([augment(xi)for xi in x])
            x,y=x.to(device,non_blocking=True),y.to(device,non_blocking=True)
            opt.zero_grad();cri(m(x),y).backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(),1.0)
            opt.step();ema.update(m)
        sch.step()
        ema.s.eval();c,t=0,0
        with torch.no_grad():
            for x,y in vl:x,y=x.to(device),y.to(device);c+=(ema.s(x).argmax(1)==y).sum().item();t+=y.size(0)
        acc=c/t
        if acc>ba:ba=acc;be=copy.deepcopy(ema.s.state_dict());pa=0
        else:pa+=1
        if pa>=PATIENCE:break
    ema.s.load_state_dict(be);ema.s.eval()
    with torch.no_grad():probs=torch.cat([torch.softmax(ema.s(x.to(device)),1).cpu() for x,_ in tel]).numpy()
    pr=probs.argmax(1)
    sa=accuracy_score(y_te_t,pr);f1=f1_score(y_te_t,pr,average='weighted');cm=confusion_matrix(y_te_t,pr)
    print(f'  [{name:20s}] test={sa*100:.1f}% f1={f1*100:.1f}% val={ba*100:.1f}% ep={ep+1}')
    print(f'    混淆矩阵:\n{cm}')
    return {'name':name,'acc':round(float(sa),4),'f1':round(float(f1),4),'n':len(X_tr)}

print(f'\n{"="*50}')
print('开始训练...')
print(f'{"="*50}')
rs=[]
rs.append(train('原始132张',X1_tr,y1_tr,X1_va,X1_te,y1_te))
rs.append(train('论文时间假设',X2_tr,y2_tr,X2_va,X2_te,y2_te))

print(f'\n{"="*50}')
print('结果')
print(f'{"="*50}')
print(f'{"配置":25s} {"准确率":8s} {"F1":8s} {"训练集":6s}')
for r in sorted(rs,key=lambda x:x['acc'],reverse=True):
    print(f'{r["name"]:25s} {r["acc"]*100:5.1f}% {r["f1"]*100:5.1f}% {r["n"]:4d}张')
