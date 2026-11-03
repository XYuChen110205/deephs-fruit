import os
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

from avocado_detection.config import base_dir
from avocado_detection.models import ClassifierNetwork
from avocado_detection.training import EMA, WarmupCosineScheduler, augment_batch
from avocado_detection.visualization import plot_gradcam, plot_curve, plot_lr_curve, plot_ema_compare
from avocado_detection.data import extract_features_2d_raw, diagnose_dataset
from avocado_detection.evaluation import evaluate, fruit_level_fusion


if __name__ == "__main__":

    # Set random seed for reproducibility
    seed =2
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    os.makedirs('images', exist_ok=True)

    BASE_LR          = 1e-3
    MIN_LR           = 1e-6
    WEIGHT_DECAY     = 1e-4
    BATCH_SIZE       = 32
    EPOCHS           = 150
    PATIENCE         = 30
    LR_WARMUP_EPOCHS = 10
    GRAD_CLIP        = 1.0
    EMA_WARMUP_EPOCHS = 15
    EMA_INIT_DECAY   = 0.9
    EMA_FINAL_DECAY  = 0.999
    TTA_ITER         = 5

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

    train_criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    val_criterion   = nn.CrossEntropyLoss()

    optimizer = optim.AdamW(model.parameters(), lr=BASE_LR, weight_decay=WEIGHT_DECAY)
    scheduler = WarmupCosineScheduler(optimizer, LR_WARMUP_EPOCHS, EPOCHS, BASE_LR, MIN_LR)

    ema = EMA(model, init_decay=EMA_INIT_DECAY)

    best_val_loss    = float('inf')
    best_model_state = None
    best_ema_state   = None
    train_losses, val_losses     = [], []
    train_accs,   val_accs       = [], []
    raw_val_accs,  ema_val_accs  = [], []
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
            ema.update(model)
            train_loss    += loss.item()
            train_correct += (pred.argmax(1) == y).sum().item()
            train_total   += y.size(0)

        train_losses.append(train_loss / len(train_loader))
        train_accs.append(train_correct / train_total)

        if epoch < EMA_WARMUP_EPOCHS:
            cur_ema_decay = EMA_INIT_DECAY
        else:
            ramp = min((epoch - EMA_WARMUP_EPOCHS) / max(EPOCHS - EMA_WARMUP_EPOCHS, 1), 1.0)
            cur_ema_decay = EMA_INIT_DECAY + ramp * (EMA_FINAL_DECAY - EMA_INIT_DECAY)
        ema.set_decay(cur_ema_decay)

        raw_val_loss, raw_val_acc = evaluate(model,           val_loader, val_criterion, device)
        ema_val_loss, ema_val_acc = evaluate(ema.eval_model(), val_loader, val_criterion, device)

        raw_val_accs.append(raw_val_acc)
        ema_val_accs.append(ema_val_acc)

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

    plot_curve(train_accs, val_accs,
               'Training & Validation Accuracy (v2)', 'images/avocado_accuracy_curve.png', 'Accuracy')
    plot_curve(train_losses, val_losses,
               'Training & Validation Loss (v2)',     'images/avocado_loss_curve.png',     'Loss')
    plot_lr_curve(lr_history, 'images/avocado_lr_curve.png')
    plot_ema_compare(raw_val_accs, ema_val_accs,
                     EMA_WARMUP_EPOCHS, 'images/avocado_ema_warmup_curve.png')

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
    plt.tight_layout(); plt.savefig('images/avocado_confusion_matrix.png', dpi=300); plt.close()
    print("[Sample CM] Saved → images/avocado_confusion_matrix.png")

    fruit_level_fusion(mean_probs, test_fruit_ids, y_test_enc, le,
                       target_names, 'images/avocado_confusion_matrix_fruit.png')

    plot_gradcam(inf_model, X_test_t, y_test_enc, le, device, 2, 'images/avocado_gradcam.png')

    torch.save(best_ema_state,   'avocado_hs_cnn_v2_ema.pth')
    torch.save(best_model_state, 'avocado_hs_cnn_v2_raw.pth')
    print("\nModel saved:")
    print("  avocado_hs_cnn_v2_ema.pth  ← 推荐部署")
    print("  avocado_hs_cnn_v2_raw.pth  ← 继续训练用")

    print("\n" + "="*65)
    print("输出文件汇总：")
    for fname, desc in [
        ("images/avocado_accuracy_curve.png",         "val 准确率（tracked）"),
        ("images/avocado_loss_curve.png",              "val loss（tracked）"),
        ("images/avocado_lr_curve.png",               "学习率调度曲线"),
        ("images/avocado_ema_warmup_curve.png",       "raw vs EMA val acc 对比【新增诊断图】"),
        ("images/avocado_confusion_matrix.png",       "样本级混淆矩阵"),
        ("images/avocado_confusion_matrix_fruit.png", "果实级混淆矩阵"),
        ("images/avocado_gradcam.png",                "Grad-CAM 热力图"),
        ("avocado_hs_cnn_v2_ema.pth",     "EMA 权重（部署）"),
        ("avocado_hs_cnn_v2_raw.pth",     "原始权重（训练）"),
        ("vis_2d_dataset_v2.npz",         "数据缓存"),
    ]:
        print(f"  {fname:50s} {desc}")
    print("="*65)
