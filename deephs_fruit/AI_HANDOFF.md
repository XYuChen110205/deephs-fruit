# AI 接手指南 - DeepHS-Fruit 项目

> 本文档用于快速同步项目状态给新的 AI 助手。用户已和前任 AI 完成环境搭建和稳定性调试，现在进入日常训练/实验阶段。

---

## 1. 项目基本信息

| 项目 | DeepHS-Fruit（高光谱水果成熟度识别） |
|------|--------------------------------------|
| 论文 | IJCNN 2021, Varga et al. "Measuring the Ripeness of Fruit with Hyperspectral Imaging and Deep Learning" |
| 官方仓库 | https://github.com/cogsys-tuebingen/deephs_fruit |
| **本地路径** | `E:\DeepHS-Fruit\deephs_fruit` |
| **数据路径** | `E:\DeepHS-Fruit\deephs_fruit\data`（高光谱立方体数据） |
| **当前分支** | `local-stable`（见下文说明） |
| Python 环境 | `conda activate hyperspectral`（torch 2.5.1 + lightning 2.6.1） |

---

## 2. 目录结构（关键部分）

```
E:\DeepHS-Fruit\deephs_fruit
├── classification\          # 训练入口
│   ├── train.py              # 主训练脚本（已打补丁）
│   └── model_factory.py      # 模型注册（deephs_net, hyve, resnet...）
├── core\
│   ├── datasets\
│   │   └── hyperspectral_dataset.py   # 数据集类（含磁盘缓存、NaN清洗）
│   ├── measurements\         # 数据记录列表（avocado/kiwi等清单）
│   ├── name_convention.py    # Fruit, CameraType, ClassificationType 枚举
│   └── spectral_io.py        # 波段定义（VIS:224, NIR:252, VIS_COR:249）
├── data\                     # 高光谱数据（用户已放置）
│   └── avocado\              # 牛油果数据
│       ├── VIS\              # Specim FX10（224波段）
│       ├── NIR\              # RedEye（252波段）
│       └── VIS_COR\          # Corning（249波段）
├── TRAIN_LOCAL.md            # 本地路径速查（由前任AI创建）
└── AI_HANDOFF.md             # 本文档
```

---

## 3. 分支说明（重要！）

| 分支 | 用途 | 当前状态 |
|------|------|----------|
| `master` | 接近上游原版 | 保留作参照 |
| **`local-stable`** | **日常训练用** | ✅ **当前在此分支** |

### `local-stable` 分支的改动（稳定性补丁）

**改动原则**：不动模型结构、不动数据划分逻辑，只做数值稳定性修复。

1. **`classification/train.py`**
   - Windows UTF-8 输出编码（防止 `gbk` 编码崩溃）
   - `gradient_clip_val=1.0`（防梯度爆炸 → NaN）

2. **`core/datasets/hyperspectral_dataset.py`**
   - `_preload_data()`: 逐条进度显示 + 磁盘缓存（`data\_preload_cache\v2_*`）
   - `prepare_fruit()`: NaN/Inf 清洗（`nan_to_num` + 非有限值置零 + `[0,10]`裁剪）

3. **新增文档**
   - `TRAIN_LOCAL.md`: 本地路径和快速命令参考

---

## 4. 数据集详情

### 牛油果数据（已就位）

| 相机 | 代码名 | 波段数 | 数据位置 |
|------|--------|--------|----------|
| Specim FX10 | `vis` | 224 | `data\avocado\VIS\day_*\` |
| INNO-SPEC RedEye | `nir` | 252 | `data\avocado\NIR\day_*\` |
| Corning microHSI | `vis_cor` | 249 | `data\avocado\VIS_COR\day_*\` |

### 数据格式
- ENVI 高光谱格式（`.hdr` + `.raw`）
- `spectral` 库读取（会报 `NaNValueWarning`，属正常现象，已在 `prepare_fruit` 处理）

### 任务类型（务必显式指定！）

```python
# classification/train.py 默认是 FIRMNESS！
# 成熟度实验必须显式写 ripeness
--classification_type ripeness   # 成熟度（3类：Unripe/Perfect/Overripe）
--classification_type firmness   # 硬度（默认，但 avocado 一般不用）
--classification_type sugar      # 糖度（avocado 无此标签！会报错退出）
```

---

## 5. 训练命令模板

### 基础命令结构

```powershell
cd E:\DeepHS-Fruit\deephs_fruit
$env:PYTHONPATH = (Get-Location).Path
conda activate hyperspectral

python classification/train.py `
  --data_path E:\DeepHS-Fruit\deephs_fruit\data `
  --model deephs_net `           # 或 hyve
  --fruit avocado `
  --classification_type ripeness `   # ⚠️ 必须显式写！
  --camera_type vis `            # 或 nir, vis_cor
  --seed 23312323 `
  --batch_size 64 `             # 默认64，OOM时改32
  --lr 0.001                     # 原版是0.01，易NaN，建议0.001
```

### 已验证可行的配置

**VIS + 牛油果 + 成熟度**（主实验）：
```powershell
python classification/train.py --data_path E:\DeepHS-Fruit\deephs_fruit\data --model deephs_net --fruit avocado --classification_type ripeness --camera_type vis --seed 23312323 --lr 0.001
```

**NIR + 牛油果 + 成熟度**（样本少，test仅9条）：
```powershell
python classification/train.py --data_path E:\DeepHS-Fruit\deephs_fruit\data --model deephs_net --fruit avocado --classification_type ripeness --camera_type nir --seed 23312323 --lr 0.001
```

---

## 6. 已知坑点与解决方案

### ❌ NaN loss → 训练中断

**原因**：梯度爆炸 或 输入数据含 NaN/Inf

**已处理**：
- ✅ `prepare_fruit` 里 `nan_to_num` + 清洗
- ✅ `gradient_clip_val=1.0`
- ✅ 建议 `--lr 0.001`（原版 0.01 在 Windows/RTX4060 上易炸）

### ⏱️ 预加载很慢（每次启动几分钟）

**已处理**：
- ✅ 磁盘缓存 `data\_preload_cache\v2_64x64\*.npy`
- ✅ 首次 `src: build`，后续 `src: cache`（秒级加载）

### 🎲 Test accuracy 波动大

**原因**：test 集很小（VIS ~80条，NIR 仅9条）

**建议**：
- 多 seed 跑（23312323, 42, 12345），看均值±方差
- 不要过度解读单次 test 结果

### 📁 `classification_type` 默认值陷阱

```python
# train.py 里默认是 FIRMNESS！
parser.add_argument('--classification_type', default=ClassificationType.FIRMNESS)
```

**后果**：想跑成熟度但忘写 `--classification_type ripeness`，结果跑了硬度任务，数据划分/标签全错。

**务必检查**：命令里显式写了 `ripeness` 还是 `firmness`。

---

## 7. 用户目标（请 AI 协助推进）

1. **短期**：跑通 VIS/NIR 牛油果成熟度基线，得到可复现的 accuracy
2. **中期**：多 seed 统计，和论文 0.93 accuracy 对比（注意环境差异声明）
3. **长期**：尝试 hyve 模型、其他水果（kiwi）、其他任务（firmness/sugar）

---

## 8. 快速诊断命令

```powershell
# 检查当前分支
git branch

# 查看改动（与 master 对比）
git diff master -- classification/train.py
git diff master -- core/datasets/hyperspectral_dataset.py

# 检查数据集是否存在
test-path E:\DeepHS-Fruit\deephs_fruit\data\avocado\VIS\day_01

# 检查缓存目录大小
du -sh E:\DeepHS-Fruit\deephs_fruit\data\_preload_cache  # 或 Get-ChildItem ... | Measure-Object
```

---

## 9. 交接 Checklist（新 AI 确认）

- [ ] 已读取 `AI_HANDOFF.md` 和 `TRAIN_LOCAL.md`
- [ ] 已确认当前分支是 `local-stable`
- [ ] 已确认数据路径 `E:\DeepHS-Fruit\deephs_fruit\data` 存在且含 avocado/{VIS,NIR}
- [ ] 已知悉 `--classification_type` 默认陷阱
- [ ] 已知悉 `--lr 0.001` 和 `gradient_clip` 的背景

---

**文档版本**：2024-06（由前任 AI 创建）
**如有疑问**：问用户要具体错误日志或训练输出片段
