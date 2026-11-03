"""
运行所有对比实验并生成汇总报告
对比方法:
- MLP (多层感知机)
- SVM (支持向量机)
- Random Forest (随机森林)
"""
import os
import sys
import subprocess
import json
import pandas as pd
from datetime import datetime

experiments = [
    ('mlp.py', 'MLP (多层感知机)'),
    ('svm.py', 'SVM (支持向量机)'),
    ('random_forest.py', 'Random Forest (随机森林)'),
]

results_summary = []

print('='*60)
print('开始运行对比实验')
print('='*60)

for script, description in experiments:
    print(f'\n{"="*60}')
    print(f'运行: {description}')
    print(f'脚本: {script}')
    print('='*60)
    
    if not os.path.exists(script):
        print(f'警告: 脚本不存在 {script}，跳过')
        continue
    
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
                    '模型': description,
                    '样本准确率': f"{data.get('sample_acc', 0):.4f}",
                    '水果准确率': f"{data.get('fruit_acc', 0):.4f}"
                })
        else:
            print(f'警告: 未找到结果文件 {result_file}')
            results_summary.append({
                '模型': description,
                '样本准确率': 'N/A',
                '水果准确率': 'N/A'
            })
            
    except subprocess.TimeoutExpired:
        print(f'错误: {script} 运行超时')
        results_summary.append({
            '模型': description,
            '样本准确率': 'Timeout',
            '水果准确率': 'Timeout'
        })
    except Exception as e:
        print(f'错误: 运行 {script} 时发生异常: {e}')
        results_summary.append({
            '模型': description,
            '样本准确率': 'Error',
            '水果准确率': 'Error'
        })

# 生成汇总表格
print('\n' + '='*60)
print('对比实验结果汇总')
print('='*60)

df = pd.DataFrame(results_summary)
print(df.to_string(index=False))

# 保存为CSV
csv_file = 'comparison_results_summary.csv'
df.to_csv(csv_file, index=False, encoding='utf-8-sig')
print(f'\n结果已保存到: {csv_file}')

# 保存为Markdown
md_file = 'comparison_results_summary.md'
with open(md_file, 'w', encoding='utf-8') as f:
    f.write('# 对比实验结果汇总\n\n')
    f.write(f'生成时间: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}\n\n')
    f.write('## 方法对比\n\n')
    f.write(df.to_markdown(index=False))
    f.write('\n\n## 说明\n\n')
    f.write('- **MLP**: 多层感知机，使用展平的高光谱数据\n')
    f.write('- **SVM**: 支持向量机，使用展平的高光谱数据\n')
    f.write('- **Random Forest**: 随机森林，使用展平的高光谱数据\n')
    f.write('- **样本准确率**: 单张图像级别的分类准确率\n')
    f.write('- **水果准确率**: 水果级别的分类准确率（同水果多次测量取平均）\n')

print(f'结果已保存到: {md_file}')
