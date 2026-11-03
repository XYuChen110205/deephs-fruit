"""
终极版实验 — 融合所有好技巧 + BTA
合并自：你队友的代码（WeightedSampler, LabelSmoothing, CosineAnnealing, EMA Warmup）
      + 你的代码（BTA双向时间假设, 随机划分ID防数据泄露）
"""
import os, sys, copy, random, json, time
import numpy as np
import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, WeightedRandomSampler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from collections import Counter, defaultdict

SEED = 23312323
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'设备: {device}')

sys.path.insert(0, r'e:\DeepHS-Fruit\deephs_fruit')
from core.name_convention import *
from core.measurements import all_fruits_new
from core import fruit_list
import spectral

data_path = r'e:\DeepHS-Fruit\deephs_fruit\data'

# ===== 天数排序 =====
day_order = ['day_01','day_02','day_03','day_04','day_05','day_06','day_07',
    'day_08','day_09','day_10','day_m2_01','day_m2_02','day_m2_03',
    'day_m2_04','day_m2_05','day_m2_06','day_m2_07','day_m2_08',
    'day_m2_09','day_m2_10','day_m2_11','day_m2_12','day_m2_13',
    'day_m2_14','day_m4_01','day_m4_02','day_m4_03','day_m4_04',
    'day_m4_05','day_m4_06','day_m4_07','day_m4_08','day_m4_09']
day_rank = {d:i for i,d in enumerate(day_order)}

# ===== BTA（双向时间假设）=====
def apply_bta(records):
    id_to_recs = defaultdict(list)
    for r in records: id_to_recs[r.id].append(r)
    extended = []
    for fid, recs in id_to_recs.items():
        recs = sorted(recs, key=lambda r: day_rank.get(r.day.value, 999))
        labeled = [r for r in recs if r.is_labeled()]
        if not labeled: extended.extend(recs); continue
        # 合并中间成熟度
        for r in recs:
            if r.is_labeled():
                if r.label.ripeness_state == RipenessState.RIPE:
                    r.label.ripeness_state = RipenessState.PERFECT
                elif r.label.ripeness_state == RipenessState.NEAR_OVERRIPE:
                    r.label.ripeness_state = RipenessState.OVERRIPE
        sl = sorted([r for r in recs if r.is_labeled()], key=lambda r: day_rank.get(r.day.value, 999))
        first_u = next((r for r in sl if r.label.ripeness_state == RipenessState.UNRIPE), None)
        last_o = next((r for r in reversed(sl) if r.label.ripeness_state == RipenessState.OVERRIPE), None)
        perfect = next((r for r in sl if r.label.ripeness_state == RipenessState.PERFECT), None)
        for r in recs:
            if r.is_labeled(): extended.append(r); continue
            rd = day_rank.get(r.day.value, 500)
            if first_u and rd < day_rank.get(first_u.day.value, 0):
                nl = copy.copy(first_u.label)
                nl.ripeness_state = RipenessState.UNRIPE; nl.comment = 'BTA:unripe'
                r.label = nl
            elif last_o and rd > day_rank.get(last_o.day.value, 999):
                nl = copy.copy(last_o.label)
                nl.ripeness_state = RipenessState.OVERRIPE; nl.comment = 'BTA:overripe'
                r.label = nl
            elif first_u and last_o and perfect:
                fu = day_rank.get(first_u.day.value, 0)
                lo = day_rank.get(last_o.day.value, 999)
                if fu < rd < lo:
                    nl = copy.copy(perfect.label)
                    nl.ripeness_state = RipenessState.PERFECT; nl.comment = 'BTA:perfect'
                    r.label = nl
            extended.append(r)
    return extended

# ===== 获取数据 =====
print('\n获取完整 avocado VIS 数据...')
full_set = fruit_list.get_for_camera_type(
    fruit_list.get_for_fruit(all_fruits_new, Fruit.AVOCADO), CameraType.VIS)
print(f'总计: {len(full_set)}, 有标注: {sum(1 for r in full_set if r.is_labeled())}')

# ===== 3 种数据集配置 =====
configs = []

# 配置A: 原始硬编码（v1 92张）
print('\n--- 配置A: 原始硬编码 (v1 92张) ---')
from core.datasets.hyperspectral_dataset import get_records
tr, va, te = get_records(Fruit.AVOCADO, CameraType.VIS, ClassificationType.RIPENESS,
    use_inter_ripeness_levels=True, extend_by_time_assumption=False, random_split=False)

def load_split(records, data_path):
    X,y,fids = [],[],[]
    for r in records:
        if not r.is_labeled(): continue
        try:
            p=r.get_file_path(); h=os.path.join(data_path,p+'.hdr'); b=os.path.join(data_path,p+'.bin')
            if not os.path.exists(h): h=os.path.join(data_path,p.replace('Avocado','avocado')+'.hdr'); b=os.path.join(data_path,p.replace('Avocado','avocado')+'.bin')
            img=spectral.envi.open(h,image=b).load()
            img=np.nan_to_num(img,nan=0.0,posinf=0.0,neginf=0.0); img=np.clip(img,-1e10,1e10)
            t=torch.tensor(img,dtype=torch.float32).permute(2,0,1).unsqueeze(0)
            t=nn.functional.interpolate(t,size=(62,62),mode='bilinear',align_corners=False).squeeze(0).numpy()
            p2=np.zeros((224,64,64),dtype=np.float32); p2[:,1:-1,1:-1]=t
            lm={RipenessState.UNRIPE:'unripe',RipenessState.PERFECT:'perfect',RipenessState.OVERRIPE:'overripe'}
            y.append(lm[r.label.ripeness_state]); X.append(p2); fids.append(str(r.id))
        except: continue
    return np.array(X),np.array(y),np.array(fids)

print('加载数据A...')
X_tr_a,y_tr_a,_=load_split(tr,data_path); X_va_a,y_va_a,_=load_split(va,data_path); X_te_a,y_te_a,te_fids=load_split(te,data_path)
print(f'  Train: {len(X_tr_a)} {dict(Counter(y_tr_a))}')
print(f'  Val:   {len(X_va_a)} {dict(Counter(y_va_a))}')
print(f'  Test:  {len(X_te_a)} {dict(Counter(y_te_a))}')
configs.append(('A-原始硬编码92张',X_tr_a,y_tr_a,X_va_a,y_va_a,X_te_a,y_te_a))

# 配置B: 随机划分ID + 论文时间假设（240张）
print('\n--- 配置B: 随机划分ID+论文时间假设 ---')
tr2,va2,te2=get_records(Fruit.AVOCADO,CameraType.VIS,ClassificationType.RIPENESS,
    use_inter_ripeness_levels=True,extend_by_time_assumption=True,random_split=True,seed=SEED)
print('加载数据B...')
X_tr_b,y_tr_b,_=load_split(tr2,data_path); X_va_b,y_va_b,_=load_split(va2,data_path); X_te_b,y_te_b,_=load_split(te2,data_path)
print(f'  Train: {len(X_tr_b)} {dict(Counter(y_tr_b))}')
configs.append(('B-论文时间假设240张',X_tr_b,y_tr_b,X_va_b,y_va_b,X_te_b,y_te_b))

# 配置C: BTA扩展 + 随机划分ID
print('\n--- 配置C: BTA扩展 ---')
extended=apply_bta(full_set)
labeled=[r for r in extended if r.is_labeled()]
print(f'BTA后总计: {len(labeled)} 条')
cnt=Counter(r.label.ripeness_state.name for r in labeled); print(f'  分布: {dict(cnt)}')
# 随机按ID划分 (修正版)
id2r=defaultdict(list)
for r in labeled: id2r[r.id].append(r)
ids=list(id2r.keys()); random.shuffle(ids)
# 按ID数量划分 70%/15%/15%
n_ids=len(ids); nv=int(n_ids*0.15); nt=int(n_ids*0.15)
ids_v,ids_te,ids_tr=set(ids[:nv]),set(ids[nv:nv+nt]),set(ids[nv+nt:])
tr3=[r for fid in ids_tr for r in id2r[fid]]
va3=[r for fid in ids_v for r in id2r[fid]]
te3=[r for fid in ids_te for r in id2r[fid]]
print(f'Train: {len(tr3)}IDs={len(ids_tr)} Val:{len(va3)} Test:{len(te3)}')
X_tr_c,y_tr_c,_=load_split(tr3,data_path); X_va_c,y_va_c,_=load_split(va3,data_path); X_te_c,y_te_c,_=load_split(te3,data_path)
for n,x,y in [('Train',X_tr_c,y_tr_c),('Val',X_va_c,y_va_c),('Test',X_te_c,y_te_c)]:
    print(f'  {n}: {len(x)} {dict(Counter(y))}')
configs.append(('C-BTA扩展',X_tr_c,y_tr_c,X_va_c,y_va_c,X_te_c,y_te_c))

# ===== 归一化 =====
for i in range(len(configs)):
    name,X_tr,y_tr,X_va,y_va,X_te,y_te=configs[i]
    m=np.mean(X_tr,axis=(0,2,3),keepdims=True); s=np.std(X_tr,axis=(0,2,3),keepdims=True); s[s<1e-6]=1
    for arr in [X_tr,X_va,X_te]:
        arr[:]=(arr-m)/s; np.clip(arr,-10,10,out=arr)
    configs[i]=(name,X_tr,y_tr,X_va,y_va,X_te,y_te)

# ===== 模型（带SE）=====
class SEBlock(nn.Module):
    def __init__(self,ch,r=16):
        super().__init__();mid=max(ch//r,4)
        self.gap=nn.AdaptiveAvgPool2d(1);self.fc=nn.Sequential(nn.Linear(ch,mid),nn.ReLU(True),nn.Linear(mid,ch),nn.Sigmoid())
    def forward(self,x):return x*self.fc(self.gap(x).flatten(1)).unsqueeze(-1).unsqueeze(-1)

class Net(nn.Module):
    def __init__(self,b=224,nc=3,h=[25,30,50]):
        super().__init__();k=3
        self.b1=nn.Sequential(nn.Conv2d(b,b*k,7,padding=3,groups=b),nn.Conv2d(b*k,h[0],1),nn.ReLU(True),nn.AvgPool2d(2),nn.BatchNorm2d(h[0]),SEBlock(h[0]))
        self.b2=nn.Sequential(nn.Conv2d(h[0],h[0]*k,3,padding=1,groups=h[0]),nn.Conv2d(h[0]*k,h[1],1),nn.ReLU(True),nn.AvgPool2d(2),nn.BatchNorm2d(h[1]),SEBlock(h[1]))
        self.b3=nn.Sequential(nn.Conv2d(h[1],h[1]*k,3,padding=1,groups=h[1]),nn.Conv2d(h[1]*k,h[2],1),nn.ReLU(True),nn.BatchNorm2d(h[2]))
        self.gap=nn.AdaptiveAvgPool2d(1)
        self.fc=nn.Sequential(nn.Linear(h[2],h[2]),nn.BatchNorm1d(h[2]),nn.ReLU(True),nn.Dropout(0.3),nn.Linear(h[2],nc))
        for m in self.modules():
            if isinstance(m,nn.Conv2d):nn.init.kaiming_normal_(m.weight,mode='fan_out',nonlinearity='relu')
            elif isinstance(m,nn.Linear):nn.init.normal_(m.weight,std=0.01)
    def forward(self,x):return self.fc(self.gap(self.b3(self.b2(self.b1(x)))).flatten(1))

class EMA:
    def __init__(self,m,d=0.9):
        self.d=d;self.s=copy.deepcopy(m).eval()
        for p in self.s.parameters():p.requires_grad_(False)
    def update(self,m):
        [sp.data.mul_(self.d).add_(mp.data,alpha=1-self.d) for sp,mp in zip(self.s.parameters(),m.parameters())]
        [sb.copy_(mb) for sb,mb in zip(self.s.buffers(),m.buffers())]
    def set_decay(self,d):self.d=d

def augment(x):
    for i in range(len(x)):
        if random.random()>0.5:x[i]=torch.flip(x[i],[1])
        if random.random()>0.5:x[i]=torch.flip(x[i],[2])
        if random.random()>0.5:x[i]=torch.rot90(x[i],1,[1,2])
        if random.random()>0.5:x[i]+=torch.randn_like(x[i])*0.02
    return x

def train(name,X_tr,y_tr,X_va,y_va,X_te,y_te):
    le=LabelEncoder()
    y_tr_t=le.fit_transform(y_tr);y_v_t=le.transform(y_va);y_te_t=le.transform(y_te)
    nc=len(le.classes_)
    X_tr_t=torch.FloatTensor(X_tr);X_v_t=torch.FloatTensor(X_va);X_te_t=torch.FloatTensor(X_te)
    # WeightedSampler（队友技巧）
    cc=np.bincount(y_tr_t);w=1.0/cc;s=WeightedRandomSampler(w[y_tr_t],len(w[y_tr_t]))
    tl=DataLoader(TensorDataset(X_tr_t,torch.LongTensor(y_tr_t)),32,sampler=s,pin_memory=True)
    vl=DataLoader(TensorDataset(X_v_t,torch.LongTensor(y_v_t)),32,pin_memory=True)
    tel=DataLoader(TensorDataset(X_te_t,torch.LongTensor(y_te_t)),32,pin_memory=True)
    m=Net(nc=nc).to(device)
    opt=optim.AdamW(m.parameters(),lr=1e-3,weight_decay=1e-4)
    t_cri=nn.CrossEntropyLoss(label_smoothing=0.1);v_cri=nn.CrossEntropyLoss()
    ema=EMA(m,0.9)
    # CosineAnnealing（队友技巧）
    from torch.optim.lr_scheduler import CosineAnnealingLR
    sch=CosineAnnealingLR(opt,T_max=150,eta_min=1e-6)
    ba=1e9;be=None;pa=0;EPOCHS=150;WARMUP=15
    raw_accs,ema_accs=[],[]
    for ep in range(EPOCHS):
        m.train();lr=sch.get_last_lr()[0]
        for x,y in tl:
            x,y=x.to(device,non_blocking=True),y.to(device,non_blocking=True)
            x=augment(x)
            opt.zero_grad();t_cri(m(x),y).backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(),1.0)
            opt.step();ema.update(m)
        # EMA warmup（队友技巧）
        if ep<WARMUP:
            ema.set_decay(0.9)
            m.eval();c1,t1=0,0
            with torch.no_grad():
                for x,y in vl:x,y=x.to(device),y.to(device);c1+=(m(x).argmax(1)==y).sum().item();t1+=y.size(0)
            val_acc=c1/t1;val_loss=None
        else:
            ramp=min((ep-WARMUP)/(EPOCHS-WARMUP),1.0)
            ema.set_decay(0.9+ramp*0.099)
            ema.s.eval();c1,t1,c2,t2=0,0,0,0;l1,l2=0,0
            with torch.no_grad():
                for x,y in vl:
                    x,y=x.to(device),y.to(device)
                    p=ema.s(x);l1+=v_cri(p,y).item();c1+=(p.argmax(1)==y).sum().item();t1+=y.size(0)
                    p2=m(x);l2+=v_cri(p2,y).item();c2+=(p2.argmax(1)==y).sum().item();t2+=y.size(0)
            val_acc=c1/t1;val_loss=l1/len(vl)
            raw_accs.append(c2/t2);ema_accs.append(val_acc)
        # Early stopping 用 EMA val loss
        if ep>=WARMUP and val_loss is not None and val_loss<ba:
            ba=val_loss;be=copy.deepcopy(ema.s.state_dict());pa=0
        elif ep>=WARMUP:pa+=1
        if pa>=40:print(f'  Early stop ep={ep+1}');break
        if (ep+1)%10==0:
            tag='RAW' if ep<WARMUP else 'EMA'
            print(f'  ep={ep+1:3d} lr={lr:.2e} val({tag})={val_acc:.3f}')
        sch.step()
    if be is None:be=copy.deepcopy(ema.s.state_dict())
    ema.s.load_state_dict(be);ema.s.eval()
    with torch.no_grad():probs=torch.cat([torch.softmax(ema.s(x.to(device)),1).cpu() for x,_ in tel]).numpy()
    pr=probs.argmax(1)
    sa=accuracy_score(y_te_t,pr);f1=f1_score(y_te_t,pr,average='weighted');cm=confusion_matrix(y_te_t,pr)
    print(f'  [{name:20s}] test={sa*100:.1f}% f1={f1*100:.1f}%')
    return {'name':name,'acc':round(float(sa),4),'f1':round(float(f1),4),'n':len(X_tr)}

print(f'\n{"="*60}')
print('开始训练...')
print(f'{"="*60}')
results=[]
for name,X_tr,y_tr,X_va,y_va,X_te,y_te in configs:
    results.append(train(name,X_tr,y_tr,X_va,y_va,X_te,y_te))

print(f'\n{"="*60}')
print('最终结果')
print(f'{"="*60}')
print(f'{"配置":25s} {"准确率":8s} {"F1":8s} {"训练集":6s}')
for r in sorted(results,key=lambda x:x['acc'],reverse=True):
    print(f'{r["name"]:25s} {r["acc"]*100:5.1f}% {r["f1"]*100:5.1f}% {r["n"]:4d}张')
