import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


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
                 num_per_class=2, save_path='images/avocado_gradcam.png'):
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
        mid = x[0].sum(dim=0).cpu().numpy()
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


def plot_lr_curve(lrs, save_name='images/avocado_lr_curve.png'):
    plt.figure(figsize=(8, 3))
    plt.plot(lrs, color='#756bb1', marker='.', markersize=3)
    plt.title('Learning Rate Schedule', fontweight='bold', color='#333')
    plt.xlabel('Epoch', loc='right', color='#666'); plt.ylabel('LR')
    plt.gca().spines[['top', 'right', 'left']].set_visible(False)
    plt.gca().spines['bottom'].set_color('#ccc')
    plt.grid(axis='y', color='#eee'); plt.tight_layout()
    plt.savefig(save_name, dpi=300); plt.close()
    print(f"[LR Curve] Saved → {save_name}")


def plot_ema_compare(raw_accs, ema_accs, ema_start, save_name='images/avocado_ema_warmup_curve.png'):
    plt.figure(figsize=(10, 5))  # 增大高度
    plt.plot(raw_accs, label='Raw Model Val Acc', color='#3182bd', marker='.', markersize=4, alpha=0.8)
    plt.plot(ema_accs, label='EMA Model Val Acc', color='#e6550d', marker='.', markersize=4, alpha=0.8)
    plt.axvline(x=ema_start, color='#31a354', linestyle='--', linewidth=1.5,
                label=f'EMA starts (epoch {ema_start+1})')
    plt.title('Raw vs EMA Model Validation Accuracy', fontweight='bold', color='#333', pad=20)
    plt.xlabel('Epoch', loc='right', color='#666'); plt.ylabel('Val Accuracy')
    plt.legend(loc='best', frameon=True)
    plt.gca().spines[['top', 'right', 'left']].set_visible(False)
    plt.gca().spines['bottom'].set_color('#ccc')
    plt.grid(axis='y', color='#eee')
    plt.savefig(save_name, dpi=300, bbox_inches='tight'); plt.close()
    print(f"[EMA Compare] Saved → {save_name}")
