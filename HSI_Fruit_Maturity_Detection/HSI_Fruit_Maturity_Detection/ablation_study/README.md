# 消融实验 (Ablation Study)

本目录包含对 HS-CNN 模型各个组件的消融实验，用于验证 SE Block、EMA 和 TTA 各组件对模型性能的贡献。

## 实验设计

| 实验脚本 | 实验名称 | SE Block | EMA | TTA |
|---------|---------|----------|-----|-----|
| `00_ablation_none.py` | 基线模型 | ✗ | ✗ | ✗ |
| `01_ablation_se_only.py` | 仅 SE | ✓ | ✗ | ✗ |
| `02_ablation_se_ema.py` | SE + EMA | ✓ | ✓ | ✗ |
| `03_ablation_full.py` | 完整模型 | ✓ | ✓ | ✓ |

## 组件说明

- **SE Block**: Squeeze-and-Excitation 注意力模块，用于通道注意力机制
- **EMA**: Exponential Moving Average 指数移动平均，用于模型参数平滑
- **TTA**: Test-Time Augmentation 测试时增强，用于提升测试集性能

## 运行方法

### 运行单个实验

```bash
python 00_ablation_none.py      # 基线模型
python 01_ablation_se_only.py   # 仅 SE
python 02_ablation_se_ema.py    # SE + EMA
python 03_ablation_full.py      # 完整模型
```

### 运行所有实验并生成报告

```bash
python run_all_ablation.py
```

运行后会生成：
- `results_*.json` - 各实验的详细结果
- `ablation_results_summary.csv` - CSV格式的汇总表
- `ablation_results_summary.md` - Markdown格式的汇总表

## 评估指标

- **Sample-Level Accuracy**: 单张图像级别的分类准确率
- **Fruit-Level Accuracy**: 水果级别的分类准确率（同水果多次测量取平均）

## 数据文件

- `ablation_data.npz` - 预处理好的高光谱数据（224 bands, 64x64）
- `prepare_data.py` - 数据预处理脚本（如需要重新生成数据）
