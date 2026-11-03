# 快速对比实验 - 只跑关键配置
# 预计总时间: 80分钟 (2个实验 × 40分钟)

param(
    [string]$DataPath = "E:\DeepHS-Fruit\deephs_fruit\data",
    [string]$OutputDir = "E:\DeepHS-Fruit\experiment_results"
)

# 创建输出目录
if (-not (Test-Path $OutputDir)) {
    New-Item -ItemType Directory -Path $OutputDir -Force
}

$env:PYTHONPATH = "E:\DeepHS-Fruit\deephs_fruit"

cd E:\DeepHS-Fruit\deephs_fruit

Write-Host "========================================" -ForegroundColor Green
Write-Host "DeepHS-Fruit 快速对比实验" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green
Write-Host "实验1: SGD + CrossEntropy (基准)"
Write-Host "实验2: Focal Loss + AdaBound (最佳)"
Write-Host "任务: Avocado + VIS + Ripeness"
Write-Host "预计时间: 80分钟"
Write-Host "========================================" -ForegroundColor Green

$startTime = Get-Date

# 实验1: 基准配置 (SGD + CE)
Write-Host ""
Write-Host ">>> 实验1/2: SGD + CrossEntropy (基准)" -ForegroundColor Cyan
$cmd1 = "E:\miniconda3\envs\hyperspectral\python.exe classification/train.py " +
        "--data_path $DataPath " +
        "--model deephs_net --fruit avocado --classification_type ripeness --camera_type vis " +
        "--seed 23312323 --batch_size 32 --lr 0.01 --num_epochs 200 " +
        "--use_focal_loss False --use_adabound False --random_split 2>&1 | " +
        "Tee-Object -FilePath '$OutputDir\exp1_baseline.log'"

Invoke-Expression $cmd1

# 提取结果
$log1 = Get-Content "$OutputDir\exp1_baseline.log" -Raw
if ($log1 -match "test/accuracy.*?(\d+\.\d+)") { $acc1 = $matches[1] } else { $acc1 = "N/A" }
if ($log1 -match "test/loss.*?(\d+\.\d+)") { $loss1 = $matches[1] } else { $loss1 = "N/A" }
Write-Host "实验1完成: Test Acc = $acc1, Loss = $loss1" -ForegroundColor Green

# GPU冷却
Start-Sleep -Seconds 15

# 实验2: 最佳配置 (Focal + AdaBound)
Write-Host ""
Write-Host ">>> 实验2/2: Focal Loss + AdaBound (最佳)" -ForegroundColor Cyan
$cmd2 = "E:\miniconda3\envs\hyperspectral\python.exe classification/train.py " +
        "--data_path $DataPath " +
        "--model deephs_net --fruit avocado --classification_type ripeness --camera_type vis " +
        "--seed 23312323 --batch_size 32 --lr 0.01 --num_epochs 200 " +
        "--use_focal_loss True --use_adabound True --random_split 2>&1 | " +
        "Tee-Object -FilePath '$OutputDir\exp2_best.log'"

Invoke-Expression $cmd2

# 提取结果
$log2 = Get-Content "$OutputDir\exp2_best.log" -Raw
if ($log2 -match "test/accuracy.*?(\d+\.\d+)") { $acc2 = $matches[1] } else { $acc2 = "N/A" }
if ($log2 -match "test/loss.*?(\d+\.\d+)") { $loss2 = $matches[1] } else { $loss2 = "N/A" }
Write-Host "实验2完成: Test Acc = $acc2, Loss = $loss2" -ForegroundColor Green

# 生成报告
$endTime = Get-Date
$duration = $endTime - $startTime

$report = @"
========================================
DeepHS-Fruit 快速对比实验报告
========================================
时间: $startTime ~ $endTime
总耗时: $([math]::Round($duration.TotalMinutes, 1)) 分钟

实验配置:
- 水果: Avocado
- 相机: VIS (Specim FX 10, 224波段)
- 任务: Ripeness
- Epochs: 200

结果对比:
----------------------------------------
[实验1] SGD + CrossEntropy (基准)
  Test Accuracy: $acc1
  Test Loss: $loss1

[实验2] Focal Loss + AdaBound (最佳)
  Test Accuracy: $acc2
  Test Loss: $loss2

改进幅度:
  Accuracy: $(if($acc1 -ne "N/A" -and $acc2 -ne "N/A"){"+" + ([float]$acc2 - [float]$acc1).ToString("F2")} else {"N/A"})
  Loss: $(if($loss1 -ne "N/A" -and $loss2 -ne "N/A"){"-" + ([float]$loss1 - [float]$loss2).ToString("F2")} else {"N/A"})
----------------------------------------

详细日志:
  $OutputDir\exp1_baseline.log
  $OutputDir\exp2_best.log
========================================
"@

$report | Tee-Object -FilePath "$OutputDir\quick_report.txt"
Write-Host ""
Write-Host "实验完成! 报告保存: $OutputDir\quick_report.txt" -ForegroundColor Green
