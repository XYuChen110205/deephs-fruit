# 高光谱水果成熟度检测 - 自动对比实验脚本
# DeepHS-Fruit Ablation Study Script
# 运行前请确保: 1) conda环境已激活 2) PYTHONPATH已设置

param(
    [string]$DataPath = "E:\DeepHS-Fruit\deephs_fruit\data",
    [string]$OutputDir = "E:\DeepHS-Fruit\experiment_results",
    [int]$NumEpochs = 200,
    [string]$Seed = "23312323"
)

# 创建输出目录
if (-not (Test-Path $OutputDir)) {
    New-Item -ItemType Directory -Path $OutputDir -Force
}

# 实验配置
$experiments = @(
    # 配置1: 基准 (SGD + CrossEntropy)
    @{
        name = "baseline_sgd_ce"
        model = "deephs_net"
        fruit = "avocado"
        camera = "vis"
        classification = "ripeness"
        lr = 0.01
        batch = 32
        use_focal = $false
        use_adabound = $false
        hidden_layers = "25,30,50"
        random_split = $true
    },
    # 配置2: +Focal Loss
    @{
        name = "focal_loss"
        model = "deephs_net"
        fruit = "avocado"
        camera = "vis"
        classification = "ripeness"
        lr = 0.01
        batch = 32
        use_focal = $true
        use_adabound = $false
        hidden_layers = "25,30,50"
        random_split = $true
    },
    # 配置3: +AdaBound
    @{
        name = "adabound"
        model = "deephs_net"
        fruit = "avocado"
        camera = "vis"
        classification = "ripeness"
        lr = 0.01
        batch = 32
        use_focal = $false
        use_adabound = $true
        hidden_layers = "25,30,50"
        random_split = $true
    },
    # 配置4: Focal + AdaBound (最佳组合)
    @{
        name = "focal_adabound"
        model = "deephs_net"
        fruit = "avocado"
        camera = "vis"
        classification = "ripeness"
        lr = 0.01
        batch = 32
        use_focal = $true
        use_adabound = $true
        hidden_layers = "25,30,50"
        random_split = $true
    },
    # 配置5: 原始划分 (no random_split)
    @{
        name = "original_split"
        model = "deephs_net"
        fruit = "avocado"
        camera = "vis"
        classification = "ripeness"
        lr = 0.01
        batch = 32
        use_focal = $true
        use_adabound = $true
        hidden_layers = "25,30,50"
        random_split = $false
    },
    # 配置6: Firmness任务
    @{
        name = "firmness"
        model = "deephs_net"
        fruit = "avocado"
        camera = "vis"
        classification = "firmness"
        lr = 0.01
        batch = 32
        use_focal = $true
        use_adabound = $true
        hidden_layers = "25,30,50"
        random_split = $true
    }
)

Write-Host "========================================" -ForegroundColor Green
Write-Host "DeepHS-Fruit 自动对比实验" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green
Write-Host "GPU: RTX 4060 (8GB)"
Write-Host "实验数量: $($experiments.Count)"
Write-Host "预计总时间: $([math]::Round($experiments.Count * 40)) 分钟"
Write-Host "========================================" -ForegroundColor Green

# 记录开始时间
$startTime = Get-Date
$results = @()

for ($i = 0; $i -lt $experiments.Count; $i++) {
    $exp = $experiments[$i]
    $expNum = $i + 1
    
    Write-Host ""
    Write-Host "========================================" -ForegroundColor Cyan
    Write-Host "实验 $expNum / $($experiments.Count): $($exp.name)" -ForegroundColor Cyan
    Write-Host "任务: $($exp.classification), 相机: $($exp.camera)" -ForegroundColor Cyan
    Write-Host "========================================" -ForegroundColor Cyan
    
    # 构建命令
    $cmd = "python classification/train.py "
    $cmd += "--data_path $DataPath "
    $cmd += "--model $($exp.model) "
    $cmd += "--fruit $($exp.fruit) "
    $cmd += "--classification_type $($exp.classification) "
    $cmd += "--camera_type $($exp.camera) "
    $cmd += "--seed $Seed "
    $cmd += "--batch_size $($exp.batch) "
    $cmd += "--lr $($exp.lr) "
    $cmd += "--num_epochs $NumEpochs "
    $cmd += "--hidden_layers $($exp.hidden_layers) "
    
    if ($exp.use_focal) {
        $cmd += "--use_focal_loss True "
    }
    if ($exp.use_adabound) {
        $cmd += "--use_adabound True "
    }
    if ($exp.random_split) {
        $cmd += "--random_split "
    }
    
    # 保存日志
    $logFile = "$OutputDir\$($exp.name)_$($exp.classification).log"
    
    Write-Host "命令: $cmd" -ForegroundColor Yellow
    Write-Host "日志: $logFile" -ForegroundColor Yellow
    
    # 运行实验
    $expStartTime = Get-Date
    
    try {
        Invoke-Expression $cmd 2>&1 | Tee-Object -FilePath $logFile
        
        # 从日志提取结果
        $logContent = Get-Content $logFile -Raw
        
        # 提取 test/accuracy
        if ($logContent -match "test/accuracy.*?([0-9.]+)") {
            $testAcc = [float]$matches[1]
        } else {
            $testAcc = 0
        }
        
        # 提取 test/loss
        if ($logContent -match "test/loss.*?([0-9.]+)") {
            $testLoss = [float]$matches[1]
        } else {
            $testLoss = 0
        }
        
        $status = "成功"
    }
    catch {
        $testAcc = 0
        $testLoss = 0
        $status = "失败"
        Write-Host "实验失败: $_" -ForegroundColor Red
    }
    
    $expEndTime = Get-Date
    $expDuration = $expEndTime - $expStartTime
    
    # 记录结果
    $results += [PSCustomObject]@{
        实验名称 = $exp.name
        分类任务 = $exp.classification
        FocalLoss = $exp.use_focal
        AdaBound = $exp.use_adabound
        RandomSplit = $exp.random_split
        Test准确率 = $testAcc
        Test损失 = $testLoss
        状态 = $status
        耗时分钟 = [math]::Round($expDuration.TotalMinutes, 1)
    }
    
    Write-Host "实验 $expNum 完成: Test Acc = $testAcc" -ForegroundColor Green
    
    # 保存中间结果
    $results | Export-Csv -Path "$OutputDir\results.csv" -NoTypeInformation -Encoding UTF8
    
    # GPU冷却
    Write-Host "GPU冷却中..." -ForegroundColor Gray
    Start-Sleep -Seconds 10
}

# 生成最终报告
Write-Host ""
Write-Host "========================================" -ForegroundColor Green
Write-Host "所有实验完成!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green

$endTime = Get-Date
$totalDuration = $endTime - $startTime

Write-Host "总耗时: $([math]::Round($totalDuration.TotalHours, 2)) 小时" -ForegroundColor Green
Write-Host "结果保存: $OutputDir\results.csv" -ForegroundColor Green
Write-Host ""

# 打印结果表格
$results | Format-Table -AutoSize

# 保存完整报告
$report = @"
DeepHS-Fruit 对比实验报告
========================================
实验时间: $startTime
完成时间: $endTime
总耗时: $([math]::Round($totalDuration.TotalHours, 2)) 小时
GPU: RTX 4060 (8GB)

实验配置:
- 模型: DeepHS-Net (59.4K参数)
- 水果: Avocado
- 相机: VIS (Specim FX 10)
- Epochs: $NumEpochs
- Seed: $Seed

结果汇总:
========================================
"@

foreach ($r in $results) {
    $report += "$($r.实验名称) - $($r.分类任务): Acc=$($r.Test准确率), Loss=$($r.Test损失), 状态=$($r.状态)`n"
}

$report | Out-File -FilePath "$OutputDir\report.txt" -Encoding UTF8

Write-Host "完整报告: $OutputDir\report.txt" -ForegroundColor Green
