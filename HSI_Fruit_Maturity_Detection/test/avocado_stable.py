import os
import json
import copy
import numpy as np
import spectral
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import warnings
from collections import defaultdict

warnings.filterwarnings("ignore")
base_dir = r'D:\avocadoClaude'

# ─────────────────────────────────────────────────────────────
# 本版本修复的核心问题（相比 stable 版）：
#
#   【问题根因】Val Loss ≈ 1.0986 = log(3) = 三分类均匀分布熵值
#              说明验证阶段模型输出完全是随机的。
#              原因：EMA decay=0.998 太大，shadow 权重每步只
#              更新 0.2%，训练初期 shadow ≈ 随机初始化，
#              用它验证等于用随机模型，自然 acc=0.3333。
#
#   修复A：EMA Warmup（核心修复）
#          前 ema_warmup_epochs 个 epoch 直接用原始模型验证，
#          之后才切换到 EMA 模型，并从低 decay(0.9) 逐步
#          升至目标 decay(0.999)，避免早期 shadow 太"冷"。
#
#   修复B：双验证（Double Validation）
#          每 epoch 同时记录 raw 模型和 EMA 模型的 val acc，
#          early stopping 取两者中更好的 val loss，
#          并在训练结束时打印对比，便于诊断。
#
#   修复C：数据集诊断打印
#          训练开始前打印各 split 的样本数和类别分布，
#          快速确认数据是否正常加载、是否严重不平衡。
#
# 沿用的改进：
#   [稳定] Warmup + CosineAnnealing LR
#   [稳定] Gradient Clipping max_norm=1.0
#   [创新1] SEBlock 光谱注意力
#   [创新2] 标签平滑（训练专用）/ val_criterion 无平滑
#   [创新3] 果实级多视角融合
#   [创新4] Grad-CAM 可视化
#   [修复1] 归一化统计量仅从训练集计算
#   [修复2] val_criterion 无 label_smoothing
#
# 输出文件（v2_ 前缀）：
#   v2_accuracy_curve.png
#   v2_loss_curve.png
#   v2_lr_curve.png
#   v2_ema_warmup_curve.png   ← 新增：raw vs EMA val acc 对比
#   v2_confusion_matrix.png
#   v2_confusion_matrix_fruit.png
#   v2_gradcam.png
#   avocado_hs_cnn_v2_ema.pth
#   avocado_hs_cnn_v2_raw.pth
#   vis_2d_dataset_v2.npz
# ─────────────────────────────────────────────────────────────


# ══════════════════════════════════════════════════════════════
# EMA：支持 decay 动态调整（修复A 的核心）
# ══════════════════════════════════════════════════════════════
class EMA:
    """
    指数移动平均，支持运行时修改 decay。
    修复关键：训练初期使用低 decay（shadow 跟得上模型），
              随着训练稳定逐步升高 decay（shadow 更平滑）。
    """
    def __init__(self, model, init_decay=0.9):
        self.decay  = init_decay
        self.shadow = copy.deepcopy(model)
        self.shadow.eval()
        for p in self.shadow.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        for s_p, m_p in zip(self.shadow.parameters(), model.parameters()):
            s_p.data.mul_(self.decay).add_(m_p.data, alpha=1.0 - self.decay)
        for s_b, m_b in zip(self.shadow.buffers(), model.buffers()):
            s_b.copy_(m_b)

    def set_decay(self, decay):
        self.decay = decay

    def eval_model(self):
        return self.shadow


# ══════════════════════════════════════════════════════════════
# Warmup + CosineAnnealing LR 调度
# ══════════════════════════════════════════════════════════════
class WarmupCosineScheduler:
    def __init__(self, optimizer, warmup_epochs, total_epochs, base_lr, min_lr=1e-6):
        self.optimizer     = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs  = total_epochs
        self.base_lr       = base_lr
        self.min_lr        = min_lr
        self.current_epoch = 0

    def step(self):
        e = self.current_epoch
        if e < self.warmup_epochs:
            lr = self.base_lr * (e + 1) / self.warmup_epochs
        else:
            progress = (e - self.warmup_epochs) / max(
                self.total_epochs - self.warmup_epochs, 1)
            lr = self.min_lr + 0.5 * (self.base_lr - self.min_lr) * (
                1 + np.cos(np.pi * progress))
        for pg in self.optimizer.param_groups:
            pg['lr'] = lr
        self.current_epoch += 1
        return lr


# ══════════════════════════════════════════════════════════════
# 创新点1：SE 光谱注意力块
# ══════════════════════════════════════════════════════════════
class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        mid = max(channels // reduction, 4)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc  = nn.Sequential(
            nn.Linear(channels, mid), nn.ReLU(True),
            nn.Linear(mid, channels), nn.Sigmoid()
        )

    def forward(self, x):
        w = self.fc(self.gap(x).flatten(1)).unsqueeze(-1).unsqueeze(-1)
        return x * w


# ══════════════════════════════════════════════════════════════
# 网络主体
# ══════════════════════════════════════════════════════════════
class ClassifierNetwork(nn.Module):
    def __init__(self, bands, num_classes=3, hidden_layers=[25, 30, 50]):
        super().__init__()
        k = 3
        self.block1 = nn.Sequential(
            nn.Conv2d(bands, bands * k, 7, padding=3, groups=bands),
            nn.Conv2d(bands * k, hidden_layers[0], 1),
            nn.ReLU(True), nn.AvgPool2d(2),
            nn.BatchNorm2d(hidden_layers[0]),
            SEBlock(hidden_layers[0]),
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(hidden_layers[0], hidden_layers[0]*k, 3, padding=1, groups=hidden_layers[0]),
            nn.Conv2d(hidden_layers[0]*k, hidden_layers[1], 1),
            nn.ReLU(True), nn.AvgPool2d(2),
            nn.BatchNorm2d(hidden_layers[1]),
            SEBlock(hidden_layers[1]),
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(hidden_layers[1], hidden_layers[1]*k, 3, padding=1, groups=hidden_layers[1]),
            nn.Conv2d(hidden_layers[1]*k, hidden_layers[2], 1),
            nn.ReLU(True), nn.BatchNorm2d(hidden_layers[2]),
        )
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc  = nn.Sequential(
            nn.Linear(hidden_layers[2], hidden_layers[2]),
            nn.BatchNorm1d(hidden_layers[2]),
            nn.ReLU(True), nn.Dropout(0.3),          # Dropout 从 0.4 降到 0.3
            nn.Linear(hidden_layers[2], num_classes)
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None: nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                nn.init.ones_(m.weight); nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None: nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.fc(self.gap(self.block3(self.block2(self.block1(x)))).flatten(1))


# ══════════════════════════════════════════════════════════════
# 创新点4：Grad-CAM
# ══════════════════════════════════════════════════════════════
class GradCAM:
    def __init__(self, model):
        self.model = model
        self.activations = self.gradients = None
        target = list(model.block3.children())[-1]
        self._fwd = target.register_forward_hook(
            lambda m, i, o: setattr(self, 'activations', o.detach().clone()))
        self._bwd = target.register_full_backward_hook(
            lambda m, gi, go: setattr(self, 'gradients', go[0].detach().clone()))

    def generate(self, x, class_idx=None):
        self.model.eval()
        with torch.enable_grad():
            x_in = x.clone().requires_grad_(True)
            logits = self.model(x_in)
            if class_idx is None: class_idx = logits.argmax(1).item()
            self.model.zero_grad()
            logits[0, class_idx].backward()
        w   = self.gradients[0].mean(dim=[1, 2])
        cam = F.relu((w[:, None, None] * self.activations[0]).sum(0)).cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam, class_idx

    def remove(self):
        self._fwd.remove(); self._bwd.remove()


def plot_gradcam(model, X_test_t, y_test_enc, le, device,
                 num_per_class=2, save_path='v2_gradcam.png'):
    gc = GradCAM(model); model.eval()
    sel_idx, sel_cls = [], []
    for cls in range(len(le.classes_)):
        idxs = np.where(y_test_enc == cls)[0][:num_per_class]
        sel_idx.extend(idxs.tolist()); sel_cls.extend([cls]*len(idxs))
    n = len(sel_idx)
    fig, axes = plt.subplots(2, n, figsize=(3*n, 6))
    if n == 1: axes = axes.reshape(2, 1)
    im = None
    for col, (idx, cls) in enumerate(zip(sel_idx, sel_cls)):
        x = X_test_t[idx:idx+1].to(device)
        true_lbl = le.classes_[y_test_enc[idx]]
        cam, pred_cls = gc.generate(x)
        pred_lbl = le.classes_[pred_cls]

        # ====================== 这里我帮你改了 ======================
        # 原来用中间波段 → 全黑
        # 现在用 所有波段求和 → 一定能看到果实轮廓！
        mid = x[0].sum(dim=0).cpu().numpy()
        # ===========================================================

        mid = (mid - mid.min()) / (mid.max() - mid.min() + 1e-8)
        cam_up = F.interpolate(torch.tensor(cam).unsqueeze(0).unsqueeze(0).float(),
                               size=(64, 64), mode='bilinear', align_corners=False).squeeze().numpy()
        color = '#2ca02c' if true_lbl == pred_lbl else '#d62728'
        axes[0, col].imshow(mid, cmap='gray', vmin=0, vmax=1)
        axes[0, col].set_title(f'True: {true_lbl.capitalize()}\nPred: {pred_lbl.capitalize()}',
                               fontsize=8, color=color)
        axes[0, col].axis('off')
        axes[1, col].imshow(mid, cmap='gray', vmin=0, vmax=1)
        im = axes[1, col].imshow(cam_up, cmap='jet', alpha=0.55, vmin=0, vmax=1)
        axes[1, col].set_title('Grad-CAM', fontsize=8); axes[1, col].axis('off')
    if im is not None:
        fig.colorbar(im, ax=axes[1, :].tolist(), shrink=0.6, label='Attention')
    plt.suptitle('Grad-CAM Visualization (v2)', fontweight='bold', fontsize=10)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight'); plt.close()
    gc.remove(); print(f"[GradCAM] Saved → {save_path}")


# ══════════════════════════════════════════════════════════════
# 创新点3：果实级多视角融合
# ══════════════════════════════════════════════════════════════
def fruit_level_fusion(probs, fruit_ids, y_true_enc, le,
                       target_names, save_path='v2_confusion_matrix_fruit.png'):
    fruit_probs, fruit_labels = defaultdict(list), {}
    for i, fid in enumerate(fruit_ids):
        fruit_probs[fid].append(probs[i]); fruit_labels[fid] = y_true_enc[i]
    f_preds, f_labels = [], []
    for fid in fruit_probs:
        f_preds.append(np.stack(fruit_probs[fid]).mean(0).argmax())
        f_labels.append(fruit_labels[fid])
    f_preds, f_labels = np.array(f_preds), np.array(f_labels)
    acc = accuracy_score(f_labels, f_preds)
    print(f"\n{'='*55}")
    print(f"[Fruit-Level Fusion]  Fruits: {len(fruit_probs)}  Accuracy: {acc:.4f}")
    print('='*55)
    print(classification_report(f_labels, f_preds, target_names=target_names))
    cm = confusion_matrix(f_labels, f_preds, normalize='true')
    order = [2, 1, 0]; cm = cm[order][:, order]
    pnames = [target_names[i] for i in order]
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='.2f', cmap='Greens',
                xticklabels=pnames, yticklabels=pnames, vmin=0, vmax=1)
    plt.title('Fruit-Level Confusion Matrix (v2)', fontweight='bold')
    plt.ylabel('True Label'); plt.xlabel('Predicted Label')
    plt.tight_layout(); plt.savefig(save_path, dpi=300); plt.close()
    print(f"[Fruit CM] Saved → {save_path}")
    return acc


# ══════════════════════════════════════════════════════════════
# 工具函数
# ══════════════════════════════════════════════════════════════
def augment_batch(x):
    out = x.clone()
    for i in range(len(out)):
        if torch.rand(1) > 0.5: out[i] = torch.flip(out[i], [1])
        if torch.rand(1) > 0.5: out[i] = torch.flip(out[i], [2])
        if torch.rand(1) > 0.5: out[i] = torch.rot90(out[i], 1, [1, 2])
        if torch.rand(1) > 0.5: out[i] += torch.randn_like(out[i]) * 0.02
    return out


def plot_curve(train, val, title, save_name, ylabel=''):
    plt.figure(figsize=(8, 4))
    plt.plot(train, label='Train', color='#3182bd', marker='.', markersize=4)
    plt.plot(val,   label='Val',   color='#e6550d', marker='.', markersize=4)
    plt.title(title, fontweight='bold', color='#333')
    plt.xlabel('Epoch', loc='right', color='#666')
    if ylabel: plt.ylabel(ylabel, color='#666')
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.05), ncol=2, frameon=False)
    plt.gca().spines[['top', 'right', 'left']].set_visible(False)
    plt.gca().spines['bottom'].set_color('#ccc')
    plt.grid(axis='y', color='#eee'); plt.tight_layout()
    plt.savefig(save_name, dpi=300); plt.close()


def plot_lr_curve(lrs, save_name='v2_lr_curve.png'):
    plt.figure(figsize=(8, 3))
    plt.plot(lrs, color='#756bb1', marker='.', markersize=3)
    plt.title('Learning Rate Schedule', fontweight='bold', color='#333')
    plt.xlabel('Epoch', loc='right', color='#666'); plt.ylabel('LR')
    plt.gca().spines[['top', 'right', 'left']].set_visible(False)
    plt.gca().spines['bottom'].set_color('#ccc')
    plt.grid(axis='y', color='#eee'); plt.tight_layout()
    plt.savefig(save_name, dpi=300); plt.close()
    print(f"[LR Curve] Saved → {save_name}")


def plot_ema_compare(raw_accs, ema_accs, ema_start, save_name='v2_ema_warmup_curve.png'):
    """绘制 raw 模型 vs EMA 模型的 val acc 对比，标注 EMA 切换点"""
    plt.figure(figsize=(10, 4))
    plt.plot(raw_accs, label='Raw Model Val Acc', color='#3182bd', marker='.', markersize=4, alpha=0.8)
    plt.plot(ema_accs, label='EMA Model Val Acc', color='#e6550d', marker='.', markersize=4, alpha=0.8)
    plt.axvline(x=ema_start, color='#31a354', linestyle='--', linewidth=1.5,
                label=f'EMA starts (epoch {ema_start+1})')
    plt.title('Raw vs EMA Model Validation Accuracy', fontweight='bold', color='#333')
    plt.xlabel('Epoch', loc='right', color='#666'); plt.ylabel('Val Accuracy')
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.12), ncol=3, frameon=False)
    plt.gca().spines[['top', 'right', 'left']].set_visible(False)
    plt.gca().spines['bottom'].set_color('#ccc')
    plt.grid(axis='y', color='#eee'); plt.tight_layout()
    plt.savefig(save_name, dpi=300); plt.close()
    print(f"[EMA Compare] Saved → {save_name}")


# ══════════════════════════════════════════════════════════════
# 修复C：数据集诊断
# ══════════════════════════════════════════════════════════════
def diagnose_dataset(y_train_enc, y_val_enc, y_test_enc, le):
    """训练开始前打印各 split 的样本数和类别分布"""
    class_names = le.classes_
    print("\n" + "="*65)
    print("数据集诊断")
    print("="*65)
    for split_name, y_enc in [("Train", y_train_enc),
                               ("Val",   y_val_enc),
                               ("Test",  y_test_enc)]:
        counts = np.bincount(y_enc, minlength=len(class_names))
        total  = len(y_enc)
        print(f"\n  {split_name:5s}  总计: {total} 样本")
        for cls_idx, (name, cnt) in enumerate(zip(class_names, counts)):
            bar = '█' * int(cnt / max(counts) * 20)
            print(f"    [{cls_idx}] {name:12s}: {cnt:4d} ({cnt/total*100:5.1f}%)  {bar}")
    print("="*65 + "\n")

    # 检查潜在问题
    val_counts = np.bincount(y_val_enc, minlength=len(class_names))
    if val_counts.min() == 0:
        print("⚠️  警告：验证集中有类别样本数为 0！")
    if len(y_val_enc) < 30:
        print(f"⚠️  警告：验证集仅 {len(y_val_enc)} 个样本，acc 会大幅跳动，建议增大验证集。")
    train_counts = np.bincount(y_train_enc, minlength=len(class_names))
    imbalance = train_counts.max() / max(train_counts.min(), 1)
    if imbalance > 5:
        print(f"⚠️  警告：训练集类别不平衡比 {imbalance:.1f}x，WeightedSampler 已启用。")


# ══════════════════════════════════════════════════════════════
# 修复1：数据提取（不归一化）
# ══════════════════════════════════════════════════════════════
def extract_features_2d_raw(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    record_to_ann = {ann['record_id']: ann for ann in data.get('annotations', [])}
    X_vis, y_vis, fruit_ids = [], [], []
    for record in tqdm(data['records'], desc=f"Processing {os.path.basename(json_path)}"):
        record_id = record['id']
        if record_id not in record_to_ann or record['fruit'].lower() != 'avocado': continue
        label = record_to_ann[record_id]['ripeness_state']
        if record['camera_type'] != 'VIS': continue
        hdr_path = os.path.join(base_dir, record['files']['header_file'])
        bin_path = os.path.join(base_dir, record['files']['data_file'])
        if not os.path.exists(hdr_path) or not os.path.exists(bin_path): continue
        try:
            img = spectral.envi.open(hdr_path, image=bin_path).load()
            img = np.nan_to_num(img)
            if img.ndim < 3: continue
            t = torch.tensor(img, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
            t = F.interpolate(t, size=(62, 62), mode='bilinear', align_corners=False).squeeze(0).numpy()
            resized = np.zeros((t.shape[0], 64, 64), dtype=np.float32)
            resized[:, 1:-1, 1:-1] = t
            fruit_id = os.path.basename(hdr_path).split('_')[-2]
            X_vis.append(resized); y_vis.append(label); fruit_ids.append(fruit_id)
        except Exception: continue
    return np.array(X_vis), np.array(y_vis), np.array(fruit_ids)


# ══════════════════════════════════════════════════════════════
# 验证函数（支持任意模型，无增强）
# ══════════════════════════════════════════════════════════════
@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        pred = model(x)
        total_loss += criterion(pred, y).item()
        correct    += (pred.argmax(1) == y).sum().item()
        total      += y.size(0)
    return total_loss / len(loader), correct / total


# ══════════════════════════════════════════════════════════════
# 主流程
# ══════════════════════════════════════════════════════════════
if __name__ == "__main__":

    # ── 超参数 ──────────────────────────────────────────────────
    BASE_LR          = 1e-3
    MIN_LR           = 1e-6
    WEIGHT_DECAY     = 1e-4
    BATCH_SIZE       = 32
    EPOCHS           = 150
    PATIENCE         = 30
    LR_WARMUP_EPOCHS = 10      # LR warmup 轮数
    GRAD_CLIP        = 1.0
    EMA_WARMUP_EPOCHS = 15     # 【修复A】前15 epoch 用 raw 模型验证
    EMA_INIT_DECAY   = 0.9     # 【修复A】EMA 起始 decay（低，跟得上模型）
    EMA_FINAL_DECAY  = 0.999   # 【修复A】EMA 目标 decay（高，平滑）
    TTA_ITER         = 5
    # ────────────────────────────────────────────────────────────

    cache_path = "vis_2d_dataset_v2.npz"

    if os.path.exists(cache_path):
        print("Loading cached dataset...")
        d = np.load(cache_path, allow_pickle=True)
        X_train_vis    = d['X_train_vis'];  y_train_vis    = d['y_train_vis']
        X_val_vis      = d['X_val_vis'];    y_val_vis      = d['y_val_vis']
        X_test_vis     = d['X_test_vis'];   y_test_vis     = d['y_test_vis']
        test_fruit_ids = d['test_fruit_ids']
        train_mean     = d['train_mean'];   train_std      = d['train_std']
    else:
        print("Extracting raw features...")
        X_train_raw, y_train_vis, _ = extract_features_2d_raw(
            os.path.join(base_dir, 'annotations', 'train_only_labeled.json'))
        X_val_raw,   y_val_vis,   _ = extract_features_2d_raw(
            os.path.join(base_dir, 'annotations', 'val.json'))
        X_test_raw,  y_test_vis, test_fruit_ids = extract_features_2d_raw(
            os.path.join(base_dir, 'annotations', 'test.json'))

        # 修复1：只用训练集计算统计量
        train_mean = X_train_raw.mean(axis=(0, 2, 3), keepdims=True)[0]
        train_std  = X_train_raw.std( axis=(0, 2, 3), keepdims=True)[0] + 1e-8
        X_train_vis = (X_train_raw - train_mean) / train_std
        X_val_vis   = (X_val_raw   - train_mean) / train_std
        X_test_vis  = (X_test_raw  - train_mean) / train_std

        np.savez(cache_path,
                 X_train_vis=X_train_vis, y_train_vis=y_train_vis,
                 X_val_vis=X_val_vis,     y_val_vis=y_val_vis,
                 X_test_vis=X_test_vis,   y_test_vis=y_test_vis,
                 test_fruit_ids=test_fruit_ids,
                 train_mean=train_mean,   train_std=train_std)

    X_train_vis = np.clip(np.nan_to_num(X_train_vis), -10, 10)
    X_val_vis   = np.clip(np.nan_to_num(X_val_vis),   -10, 10)
    X_test_vis  = np.clip(np.nan_to_num(X_test_vis),  -10, 10)

    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train_vis)
    y_val_enc   = le.transform(y_val_vis)
    y_test_enc  = le.transform(y_test_vis)

    # ── 修复C：数据集诊断 ─────────────────────────────────────────
    diagnose_dataset(y_train_enc, y_val_enc, y_test_enc, le)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    X_train_t = torch.FloatTensor(X_train_vis); y_train_t = torch.LongTensor(y_train_enc)
    X_val_t   = torch.FloatTensor(X_val_vis);   y_val_t   = torch.LongTensor(y_val_enc)
    X_test_t  = torch.FloatTensor(X_test_vis);  y_test_t  = torch.LongTensor(y_test_enc)

    class_counts = np.bincount(y_train_enc)
    weights  = 1.0 / class_counts
    sampler  = torch.utils.data.WeightedRandomSampler(weights[y_train_enc], len(weights[y_train_enc]))

    train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=BATCH_SIZE, sampler=sampler)
    val_loader   = DataLoader(TensorDataset(X_val_t,   y_val_t),   batch_size=BATCH_SIZE, shuffle=False)
    test_loader  = DataLoader(TensorDataset(X_test_t,  y_test_t),  batch_size=BATCH_SIZE, shuffle=False)

    model = ClassifierNetwork(bands=X_train_vis.shape[1], num_classes=len(le.classes_)).to(device)

    # 修复2 + 创新点2
    train_criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    val_criterion   = nn.CrossEntropyLoss()

    optimizer = optim.AdamW(model.parameters(), lr=BASE_LR, weight_decay=WEIGHT_DECAY)
    scheduler = WarmupCosineScheduler(optimizer, LR_WARMUP_EPOCHS, EPOCHS, BASE_LR, MIN_LR)

    # 修复A：EMA 从低 decay 开始
    ema = EMA(model, init_decay=EMA_INIT_DECAY)

    best_val_loss    = float('inf')
    best_model_state = None
    best_ema_state   = None
    train_losses, val_losses     = [], []
    train_accs,   val_accs       = [], []
    raw_val_accs,  ema_val_accs  = [], []   # 修复B：双验证记录
    lr_history                   = []
    patience_counter             = 0

    print(f"{'='*65}")
    print("Training v2 — 修复汇总：")
    print(f"  [修复A] EMA Warmup：前{EMA_WARMUP_EPOCHS} epoch 用 raw 模型验证，")
    print(f"          之后 EMA decay 从 {EMA_INIT_DECAY} 线性升至 {EMA_FINAL_DECAY}")
    print( "  [修复B] 双验证：每 epoch 同时记录 raw / EMA 的 val acc")
    print( "  [修复C] 训练前打印数据集诊断")
    print( "  [修复1] 归一化仅用训练集统计量")
    print( "  [修复2] val_criterion 无 label_smoothing")
    print( "  [创新1] SEBlock 光谱注意力")
    print( "  [创新2] 标签平滑（训练专用）")
    print( "  [创新3] 果实级多视角融合")
    print( "  [创新4] Grad-CAM 可视化")
    print(f"{'='*65}\n")

    for epoch in range(EPOCHS):

        # ── 训练 ──────────────────────────────────────────────────
        model.train()
        cur_lr = scheduler.step(); lr_history.append(cur_lr)
        train_loss, train_correct, train_total = 0, 0, 0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            x = augment_batch(x)
            optimizer.zero_grad()
            pred = model(x)
            loss = train_criterion(pred, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRAD_CLIP)
            optimizer.step()
            ema.update(model)   # 每 step 更新 shadow
            train_loss    += loss.item()
            train_correct += (pred.argmax(1) == y).sum().item()
            train_total   += y.size(0)

        train_losses.append(train_loss / len(train_loader))
        train_accs.append(train_correct / train_total)

        # ── 修复A：动态调整 EMA decay ─────────────────────────────
        # 前 EMA_WARMUP_EPOCHS 用 raw 验证，
        # 之后 decay 从 EMA_INIT_DECAY 线性升至 EMA_FINAL_DECAY
        if epoch < EMA_WARMUP_EPOCHS:
            cur_ema_decay = EMA_INIT_DECAY
        else:
            ramp = min((epoch - EMA_WARMUP_EPOCHS) / max(EPOCHS - EMA_WARMUP_EPOCHS, 1), 1.0)
            cur_ema_decay = EMA_INIT_DECAY + ramp * (EMA_FINAL_DECAY - EMA_INIT_DECAY)
        ema.set_decay(cur_ema_decay)

        # ── 修复B：双验证 ──────────────────────────────────────────
        raw_val_loss, raw_val_acc = evaluate(model,           val_loader, val_criterion, device)
        ema_val_loss, ema_val_acc = evaluate(ema.eval_model(), val_loader, val_criterion, device)

        raw_val_accs.append(raw_val_acc)
        ema_val_accs.append(ema_val_acc)

        # early stopping 基于 raw（前 warmup）或 EMA（之后）的 val_loss
        if epoch < EMA_WARMUP_EPOCHS:
            tracked_val_loss = raw_val_loss
            tracked_val_acc  = raw_val_acc
        else:
            tracked_val_loss = ema_val_loss
            tracked_val_acc  = ema_val_acc

        val_losses.append(tracked_val_loss)
        val_accs.append(tracked_val_acc)

        if tracked_val_loss < best_val_loss:
            best_val_loss    = tracked_val_loss
            best_model_state = copy.deepcopy(model.state_dict())
            best_ema_state   = copy.deepcopy(ema.shadow.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1

        if (epoch + 1) % 5 == 0:
            ema_flag = "EMA" if epoch >= EMA_WARMUP_EPOCHS else "RAW"
            print(f"Epoch {epoch+1:3d} | LR: {cur_lr:.2e} | EMA↑{cur_ema_decay:.3f} | "
                  f"Train: {train_losses[-1]:.4f} | "
                  f"Val({ema_flag}): {tracked_val_loss:.4f} acc={tracked_val_acc:.4f} | "
                  f"Raw acc={raw_val_acc:.4f} EMA acc={ema_val_acc:.4f}")

        if patience_counter >= PATIENCE:
            print(f"Early stopping at epoch {epoch+1}.")
            break

    # ── 曲线保存 ──────────────────────────────────────────────────
    plot_curve(train_accs, val_accs,
               'Training & Validation Accuracy (v2)', 'v2_accuracy_curve.png', 'Accuracy')
    plot_curve(train_losses, val_losses,
               'Training & Validation Loss (v2)',     'v2_loss_curve.png',     'Loss')
    plot_lr_curve(lr_history, 'v2_lr_curve.png')
    plot_ema_compare(raw_val_accs, ema_val_accs,
                     EMA_WARMUP_EPOCHS, 'v2_ema_warmup_curve.png')

    # ── TTA 推理（最佳 EMA 模型）─────────────────────────────────
    ema.shadow.load_state_dict(best_ema_state)
    inf_model = ema.eval_model(); inf_model.eval()

    all_preds = torch.zeros(TTA_ITER, len(X_test_t), len(le.classes_))
    print("\nStarting TTA Inference...")
    with torch.no_grad():
        for i in range(TTA_ITER):
            preds_list = []
            for x, _ in test_loader:
                x = x.to(device)
                if i != 0:
                    x = x.clone()
                    for j in range(len(x)):
                        if torch.rand(1) > 0.5: x[j] = torch.flip(x[j], [1])
                        if torch.rand(1) > 0.5: x[j] = torch.flip(x[j], [2])
                        if torch.rand(1) > 0.5: x[j] = torch.rot90(x[j], 1, [1, 2])
                preds_list.append(F.softmax(inf_model(x), dim=1).cpu())
            all_preds[i] = torch.cat(preds_list, dim=0)

    mean_probs  = all_preds.mean(0).numpy()
    final_preds = mean_probs.argmax(1)
    final_acc   = accuracy_score(y_test_enc, final_preds)

    target_names = [n.capitalize() if n != 'perfect' else 'Ripe' for n in le.classes_]
    print(f"\n{'='*55}")
    print(f"[Sample-Level TTA + EMA]  Accuracy: {final_acc:.4f}")
    print('='*55)
    print(classification_report(y_test_enc, final_preds, target_names=target_names))

    cm = confusion_matrix(y_test_enc, final_preds, normalize='true')
    order = [2, 1, 0]; cm = cm[order][:, order]
    pnames = [target_names[i] for i in order]
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=pnames, yticklabels=pnames, vmin=0, vmax=1.0)
    plt.title('Sample-Level Confusion Matrix (v2)')
    plt.ylabel('True Label'); plt.xlabel('Predicted Label')
    plt.tight_layout(); plt.savefig('v2_confusion_matrix.png', dpi=300); plt.close()
    print("[Sample CM] Saved → v2_confusion_matrix.png")

    fruit_level_fusion(mean_probs, test_fruit_ids, y_test_enc, le,
                       target_names, 'v2_confusion_matrix_fruit.png')

    plot_gradcam(inf_model, X_test_t, y_test_enc, le, device, 2, 'v2_gradcam.png')

    torch.save(best_ema_state,   'avocado_hs_cnn_v2_ema.pth')
    torch.save(best_model_state, 'avocado_hs_cnn_v2_raw.pth')
    print("\nModel saved:")
    print("  avocado_hs_cnn_v2_ema.pth  ← 推荐部署")
    print("  avocado_hs_cnn_v2_raw.pth  ← 继续训练用")

    print("\n" + "="*65)
    print("输出文件汇总：")
    for fname, desc in [
        ("v2_accuracy_curve.png",         "val 准确率（tracked）"),
        ("v2_loss_curve.png",              "val loss（tracked）"),
        ("v2_lr_curve.png",               "学习率调度曲线"),
        ("v2_ema_warmup_curve.png",       "raw vs EMA val acc 对比【新增诊断图】"),
        ("v2_confusion_matrix.png",       "样本级混淆矩阵"),
        ("v2_confusion_matrix_fruit.png", "果实级混淆矩阵"),
        ("v2_gradcam.png",                "Grad-CAM 热力图"),
        ("avocado_hs_cnn_v2_ema.pth",     "EMA 权重（部署）"),
        ("avocado_hs_cnn_v2_raw.pth",     "原始权重（训练）"),
        ("vis_2d_dataset_v2.npz",         "数据缓存"),
    ]:
        print(f"  {fname:40s} {desc}")
    print("="*65)