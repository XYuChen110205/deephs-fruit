# 牛油果成熟度检测系统

基于高光谱成像的牛油果成熟度智能检测系统，使用深度学习实现果实级别分类。

## 项目结构

```
HSI_Fruit_Maturity_Detection/
├── avocado_detection/          # 主模块
│   ├── __init__.py             # 包入口
│   ├── config.py               # 配置
│   ├── models.py               # 网络模型
│   ├── training.py             # 训练工具
│   ├── visualization.py        # 可视化
│   ├── data.py                 # 数据处理
│   ├── evaluation.py           # 评估
│   ├── main.py                 # 主入口
│   └── images/                 # 输出图片
└── datasets/                   # 数据集
    ├── annotations/            # 标注文件
    └── avocado/                # 高光谱图像
```

## 快速开始

### 环境要求
- Python 3.8+
- PyTorch 2.0+
- NumPy, SciPy, scikit-learn, matplotlib, seaborn, spectral, tqdm

### 安装依赖
```bash
pip install torch torchvision numpy scipy scikit-learn matplotlib seaborn spectral tqdm
```

### 运行
```bash
python -m avocado_detection.main
```

## 核心功能
- 高光谱数据提取与处理
- 带 SE 注意力块的 CNN 分类
- 多视角果实级融合
- Grad-CAM 可视化
- EMA 模型优化
- 学习率调度 (Warmup + CosineAnnealing)
- 数据增强

## 输出结果

运行后在 images/ 目录生成：
- avocado_accuracy_curve.png - 准确率曲线
- avocado_loss_curve.png - 损失曲线
- avocado_lr_curve.png - 学习率曲线
- avocado_ema_warmup_curve.png - Raw vs EMA 对比
- avocado_confusion_matrix.png - 样本级混淆矩阵
- avocado_confusion_matrix_fruit.png - 果实级混淆矩阵
- avocado_gradcam.png - Grad-CAM 热力图

## 配置

### 数据路径
在 avocado_detection/config.py 中设置：
```python
base_dir = r'D:\HSI_Fruit_Maturity_Detection\datasets'
```

### 训练参数
在 main.py 中调整：
- BASE_LR - 基础学习率
- BATCH_SIZE - 批次大小
- EPOCHS - 训练轮数
- PATIENCE - 早停 patience
- TTA_ITER - 测试时增强次数

## 模型
- 输入: 64×64×(波段数) 高光谱图像
- 输出: 3 个成熟度类别 (未熟、成熟、过熟)
- 性能: 样本级准确率 ~95%，果实级准确率 ~98%

## 运行说明
1. 确保数据集路径正确
2. 安装所需依赖
3. 运行主脚本
4. 查看 images/ 目录的结果
