import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# 读取数据
df = pd.read_excel('e:\\DeepHS-Fruit\\实验数据.xlsx')

# 创建图表
fig, ax = plt.subplots(figsize=(12, 6))

x = range(len(df))
ax.plot(x, df['第一列'], marker='o', label='Column 1', linewidth=1.5, markersize=5, color='#2E86AB')
ax.plot(x, df['第二列'], marker='s', label='Column 2', linewidth=1.5, markersize=5, color='#A23B72')

ax.set_xlabel('Index', fontsize=12)
ax.set_ylabel('Value', fontsize=12)
ax.set_title('Experimental Data Trend', fontsize=14)
ax.legend(loc='best')
ax.grid(True, alpha=0.3)

# 添加平均线
mean1 = df['第一列'].mean()
mean2 = df['第二列'].mean()
ax.axhline(y=mean1, color='#2E86AB', linestyle='--', alpha=0.5)
ax.axhline(y=mean2, color='#A23B72', linestyle='--', alpha=0.5)

# 标记异常值（偏离均值超过2倍标准差）
std1 = df['第一列'].std()
std2 = df['第二列'].std()
for i, (v1, v2) in enumerate(zip(df['第一列'], df['第二列'])):
    if abs(v1 - mean1) > 2 * std1:
        ax.annotate('%.3f' % v1, (i, v1), textcoords="offset points", xytext=(0,10), ha='center', fontsize=8, color='red')
    if abs(v2 - mean2) > 2 * std2:
        ax.annotate('%.3f' % v2, (i, v2), textcoords="offset points", xytext=(0,-15), ha='center', fontsize=8, color='red')

plt.tight_layout()
plt.savefig('e:\\DeepHS-Fruit\\实验数据趋势图.png', dpi=150, bbox_inches='tight')
print('Chart saved: e:\\DeepHS-Fruit\\实验数据趋势图.png')
print('Column 1 - Mean: %.4f, Std: %.4f' % (mean1, std1))
print('Column 2 - Mean: %.4f, Std: %.4f' % (mean2, std2))

# 找出潜在异常点
outliers1 = [(i, v) for i, v in enumerate(df['第一列']) if abs(v - mean1) > 2 * std1]
outliers2 = [(i, v) for i, v in enumerate(df['第二列']) if abs(v - mean2) > 2 * std2]
if outliers1:
    print('Column 1 potential outliers:', outliers1)
if outliers2:
    print('Column 2 potential outliers:', outliers2)
