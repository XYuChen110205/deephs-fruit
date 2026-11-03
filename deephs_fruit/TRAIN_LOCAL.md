# 本机路径说明（E:\DeepHS-Fruit\deephs_fruit）

本仓库已同步若干**本地稳定性改动**（与上游 zip/clone 可能不一致）：`core/datasets/hyperspectral_dataset.py`（预加载进度 + `data\_preload_cache\v2_*` 磁盘缓存 + `prepare_fruit` NaN/Inf 清洗）、`classification/train.py`（Windows UTF-8 控制台、`gradient_clip_val=1.0`）。

- **源码根目录**：`E:\DeepHS-Fruit\deephs_fruit`
- **训练时 `--data_path` 应指向数据根目录**（下面要有 `Avocado\VIS\...` 这类相对路径）：  
  **`E:\DeepHS-Fruit\deephs_fruit\data`**

当前数据布局已核对：`data\avocado\` 下含 `VIS`、`NIR`、`VIS_COR` 及各 `day_*` 目录。

## PowerShell 示例（VIS + 牛油果 + 成熟度）

```powershell
cd E:\DeepHS-Fruit\deephs_fruit
$env:PYTHONPATH = (Get-Location).Path
# 使用你已配置好的环境，例如：
# conda activate hyperspectral
python classification/train.py `
  --data_path E:\DeepHS-Fruit\deephs_fruit\data `
  --model deephs_net `
  --fruit avocado `
  --classification_type ripeness `
  --camera_type vis `
  --seed 23312323
```

注意：`train.py` 里默认 `--classification_type` 可能是 **firmness**，复现 README 的成熟度实验时必须**显式写 `ripeness`**。

## NIR 示例

将 `--camera_type vis` 改为 `--camera_type nir` 即可。

## 若你有 `run_training_robust.py`（仅部分克隆带此脚本）

```powershell
cd E:\DeepHS-Fruit\deephs_fruit
$env:PYTHONPATH = (Get-Location).Path
python run_training_robust.py --data-path E:\DeepHS-Fruit\deephs_fruit\data --camera-type vis
```

（参数以你仓库里脚本的 argparse 为准。）
