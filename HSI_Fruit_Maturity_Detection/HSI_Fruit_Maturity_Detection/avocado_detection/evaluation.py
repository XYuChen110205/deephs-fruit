import torch
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict


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


def fruit_level_fusion(probs, fruit_ids, y_true_enc, le,
                       target_names, save_path='images/avocado_confusion_matrix_fruit.png'):
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
