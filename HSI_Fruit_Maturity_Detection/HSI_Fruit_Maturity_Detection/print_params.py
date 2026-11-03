#!/usr/bin/env python3
"""打印模型真实参数量 — 用于竞赛报告数据核实"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from avocado_detection.models import ClassifierNetwork, SEBlock
import torch

def count_params(model, label=""):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"{'='*60}")
    if label:
        print(f"  {label}")
    print(f"  总参数量:     {total:>10,}  ({total/1000:.1f}K)")
    print(f"  可训练参数量: {trainable:>10,}  ({trainable/1000:.1f}K)")
    return total

# ============================================================
# 1. 完整模型 (带 SE Block) — 我们实际使用的模型
# ============================================================
model_full = ClassifierNetwork(bands=224, num_classes=3, hidden_layers=[25, 30, 50])
total_full = count_params(model_full, "完整模型 (HS-CNN + SE) — 我们的模型")

# ============================================================
# 2. 逐层参数分解
# ============================================================
print(f"\n{'='*60}")
print("  逐层参数量分解:")
print(f"  {'层名称':<40s} {'参数量':>10s}")
print(f"  {'-'*55}")
for name, param in model_full.named_parameters():
    print(f"  {name:<40s} {param.numel():>10,}")

# ============================================================
# 3. 按模块统计
# ============================================================
print(f"\n{'='*60}")
print("  按模块统计:")
for block_name in ['block1', 'block2', 'block3', 'fc']:
    block = getattr(model_full, block_name)
    block_params = sum(p.numel() for p in block.parameters())
    print(f"  {block_name:<15s} {block_params:>10,}  ({block_params/1000:.1f}K)")

# ============================================================
# 4. SE Block 单独统计
# ============================================================
print(f"\n{'='*60}")
print("  SE Block 参数量:")
se_total = 0
for name, module in model_full.named_modules():
    if isinstance(module, SEBlock):
        se_params = sum(p.numel() for p in module.parameters())
        se_total += se_params
        print(f"  {name:<30s} {se_params:>8,}")
print(f"  {'SE总计':<30s} {se_total:>8,}")

# ============================================================
# 5. 无SE的Baseline模型参数量 (用 IdentityBlock 替代 SE)
# ============================================================
import torch.nn as nn

class IdentityBlock(nn.Module):
    def forward(self, x): return x

class BaselineNetwork(nn.Module):
    def __init__(self, bands, num_classes=3, hidden_layers=[25, 30, 50]):
        super().__init__()
        k = 3
        self.block1 = nn.Sequential(
            nn.Conv2d(bands, bands*k, 7, padding=3, groups=bands),
            nn.Conv2d(bands*k, hidden_layers[0], 1),
            nn.ReLU(True), nn.AvgPool2d(2),
            nn.BatchNorm2d(hidden_layers[0]),
            IdentityBlock(),
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(hidden_layers[0], hidden_layers[0]*k, 3, padding=1, groups=hidden_layers[0]),
            nn.Conv2d(hidden_layers[0]*k, hidden_layers[1], 1),
            nn.ReLU(True), nn.AvgPool2d(2),
            nn.BatchNorm2d(hidden_layers[1]),
            IdentityBlock(),
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(hidden_layers[1], hidden_layers[1]*k, 3, padding=1, groups=hidden_layers[1]),
            nn.Conv2d(hidden_layers[1]*k, hidden_layers[2], 1),
            nn.ReLU(True), nn.BatchNorm2d(hidden_layers[2]),
        )
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(hidden_layers[2], hidden_layers[2]),
            nn.BatchNorm1d(hidden_layers[2]),
            nn.ReLU(True), nn.Dropout(0.3),
            nn.Linear(hidden_layers[2], num_classes)
        )

baseline = BaselineNetwork(bands=224, num_classes=3)
total_baseline = count_params(baseline, "Baseline模型 (无SE)")

print(f"\n{'='*60}")
print(f"  SE 额外增加参数: {total_full - total_baseline:,}")
print(f"  SE 占比:         {(total_full - total_baseline)/total_full*100:.2f}%")

# ============================================================
# 6. 模型大小估算
# ============================================================
dummy_path = "__temp_model_size_check.pth"
torch.save(model_full.state_dict(), dummy_path)
file_size = os.path.getsize(dummy_path)
os.remove(dummy_path)
print(f"\n{'='*60}")
print(f"  模型文件大小:    {file_size:,} bytes ({file_size/1024:.1f} KB)")
print(f"{'='*60}")

# ============================================================
# 7. 与 ResNet-18 对比
# ============================================================
try:
    from torchvision.models import resnet18
    r18 = resnet18(num_classes=3)
    r18_params = sum(p.numel() for p in r18.parameters())
    print(f"\n  对比: ResNet-18 参数量 = {r18_params:,} ({r18_params/1e6:.1f}M)")
    print(f"  本文模型 / ResNet-18  = 1/{r18_params//total_full}")
except ImportError:
    print("  (torchvision 未安装, 跳过 ResNet-18 对比)")

print(f"\n{'='*60}")
print("  【用于报告的关键数字】")
print(f"  完整模型参数量: {total_full:,} ({total_full/1000:.1f}K)")
print(f"  Baseline参数量: {total_baseline:,} ({total_baseline/1000:.1f}K)")
print(f"  SE额外参数:     {total_full - total_baseline:,}")
print(f"  模型文件大小:   {file_size/1024:.1f} KB")
print(f"{'='*60}")
