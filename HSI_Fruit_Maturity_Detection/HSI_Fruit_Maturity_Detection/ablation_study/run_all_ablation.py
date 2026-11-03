"""
运行所有消融实验并生成汇总报告
"""
import os
import sys
import subprocess
import json
import pandas as pd
from datetime import datetime

experiments = [
    ('00_ablation_none.py', '基线模型 (无SE, 无EMA, 无TTA)'),
    ('01_ablation_se_only.py', '仅SE'),
    ('02_ablation_se_ema.py', 'SE + EMA'),
    ('03_ablation_full.py', '完整模型 (SE + EMA + TTA)'),
]

results_summary = []

print('='*60)
print('开始运行消融实验')
print('='*60)

for script, description in experiments:
    print(f'\n{"="*60}')
    print(f'运行: {description}')
    print(f'脚本: {script}')
    print('='*60)
    
    try:
        result = subprocess.run(
            [sys.executable, script],
            capture_output=True,
            text=True,
            timeout=3600  # 1小时超时
        )
        print(result.stdout)
        if result.stderr:
            print("错误输出:", result.stderr)
        
        # 读取结果文件
        result_file = f'results_{script.replace(".py", ".json")}'
        if os.path.exists(result_file):
            with open(result_file, 'r') as f:
                data = json.load(f)
                results_summary.append({
                    '实验': description,
                    'SE': '✓' if data['components']['SE'] else '✗',
                    'EMA': '✓' if data['components']['EMA'] else '✗',
                    'TTA': '✓' if data['components']['TTA'] else '✗',
                    '样本准确率': f"{data['sample_acc']:.4f}",
                    '水果准确率': f"{data['fruit_acc']:.4f}"
                })
        else:
            print(f'警告: 未找到结果文件 {result_file}')
            
    except subprocess.TimeoutExpired:
        print(f'错误: {script} 运行超时')
    except Exception as e:
        print(f'错误: 运行 {script} 时发生异常: {e}')

# 生成汇总表格
print('\n' + '='*60)
print('消融实验结果汇总')
print('='*60)

df = pd.DataFrame(results_summary)
print(df.to_string(index=False))

# 保存为CSV
csv_file = 'ablation_results_summary.csv'
df.to_csv(csv_file, index=False, encoding='utf-8-sig')
print(f'\n结果已保存到: {csv_file}')

# 保存为Markdown
md_file = 'ablation_results_summary.md'
with open(md_file, 'w', encoding='utf-8') as f:
    f.write('# 消融实验结果汇总\n\n')
    f.write(f'生成时间: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}\n\n')
    f.write(df.to_markdown(index=False))
    f.write('\n\n## 说明\n\n')
    f.write('- **SE**: Squeeze-and-Excitation 注意力模块\n')
    f.write('- **EMA**: Exponential Moving Average 指数移动平均\n')
    f.write('- **TTA**: Test-Time Augmentation 测试时增强\n')
    f.write('- **样本准确率**: 单张图像级别的分类准确率\n')
    f.write('- **水果准确率**: 水果级别的分类准确率（同水果多次测量取平均）\n')

print(f'结果已保存到: {md_file}')
