# 对比实验 (Comparison Experiment)

本目录包含与传统机器学习方法的对比实验，用于验证深度学习模型相对于传统方法的优势。

## 实验设计

| 实验脚本 | 方法 | 说明 |
|---------|------|------|
| `mlp.py` | MLP | 多层感知机（深度学习基准） |
| `svm.py` | SVM | 支持向量机 |
| `random_forest.py` | Random Forest | 随机森林 |

## 运行方法

### 运行单个实验

```bash
python mlp.py              # 多层感知机
python svm.py              # 支持向量机
python random_forest.py    # 随机森林
```

### 运行所有对比实验并生成报告

```bash
python run_all_comparison.py
```

运行后会生成：
- `results_*.json` - 各实验的详细结果
- `comparison_results_summary.csv` - CSV格式的汇总表
- `comparison_results_summary.md` - Markdown格式的汇总表

## 方法说明

### MLP (多层感知机)
- 输入：展平的高光谱数据 (224 × 64 × 64 = 917504 维)
- 隐藏层：[512, 256, 128]
- 激活函数：ReLU
- Dropout: 0.5

### SVM (支持向量机)
- 使用展平的高光谱特征
- RBF 核函数

### Random Forest (随机森林)
- 使用展平的高光谱特征
- 100 棵决策树

## 评估指标

- **Sample-Level Accuracy**: 单张图像级别的分类准确率
- **Fruit-Level Accuracy**: 水果级别的分类准确率（同水果多次测量取平均）

## 数据文件

数据从消融实验目录加载：
- `../ablation_study/ablation_data.npz` - 预处理好的高光谱数据
