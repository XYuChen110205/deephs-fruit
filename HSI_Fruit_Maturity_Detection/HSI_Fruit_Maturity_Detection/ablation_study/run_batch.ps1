# ============================================
# HS-CNN 批量实验一键运行
# 打开 PowerShell，cd 到这个目录，然后:
#   powershell -ExecutionPolicy Bypass -File run_batch.ps1
# 然后去吃饭，回来结果就好了
# ============================================

Write-Host "========================================" -ForegroundColor Green
Write-Host "  HS-CNN 批量实验" -ForegroundColor Green
Write-Host "  预计运行时间: 2-4 小时" -ForegroundColor Yellow
Write-Host "  去吃饭吧，回来结果就出来了" -ForegroundColor Yellow
Write-Host "========================================" -ForegroundColor Green
Write-Host ""

$startTime = Get-Date

# 设置路径
Set-Location "E:\DeepHS-Fruit\HSI_Fruit_Maturity_Detection\HSI_Fruit_Maturity_Detection\ablation_study"
$env:PYTHONPATH = "E:\DeepHS-Fruit\deephs_fruit"

Write-Host "开始时间: $startTime" -ForegroundColor Cyan
Write-Host ""

# 运行主脚本
& "E:\miniconda3\envs\hyperspectral\python.exe" run_all_experiments.py

$endTime = Get-Date
$duration = $endTime - $startTime

Write-Host ""
Write-Host "========================================" -ForegroundColor Green
Write-Host "  全部完成！" -ForegroundColor Green
Write-Host "  开始: $startTime" -ForegroundColor Cyan
Write-Host "  结束: $endTime" -ForegroundColor Cyan
Write-Host "  耗时: $($duration.Hours)小时 $($duration.Minutes)分钟" -ForegroundColor Cyan
Write-Host "  结果保存在: ablation_comparison_results.json" -ForegroundColor White
Write-Host "========================================" -ForegroundColor Green

# 打印结果摘要
Write-Host ""
Write-Host "结果摘要:" -ForegroundColor Yellow
& "E:\miniconda3\envs\hyperspectral\python.exe" -c "
import json
d = json.load(open('ablation_comparison_results.json'))
print()
print(f'最佳配置: {d[\"best\"][\"name\"]}')
print(f'最佳准确率: {d[\"best\"][\"sample_acc\"]*100:.1f}%')
print()
print(f'{\"配置\":50s} {\"准确率\":8s} {\"F1\":8s} {\"训练集\":6s}')
print('-'*72)
for r in sorted(d['results'], key=lambda x: x['sample_acc'], reverse=True):
    print(f'{r[\"name\"]:50s} {r[\"sample_acc\"]*100:5.1f}%  {r[\"f1\"]*100:5.1f}%  {r[\"n_train\"]:4d}张')
"

Read-Host "按 Enter 退出"
