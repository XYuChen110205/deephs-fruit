import re

# 读取文件
with open('DeepHS-Net-env.html', 'r', encoding='utf-8') as f:
    content = f.read()

# 定义emoji替换规则
emoji_map = {
    '🍎': '',  # 标题
    '📑': '',  # 文档目录
    '📖': '',  # 项目简介
    '🎯': '',  # 核心特性
    '📚': '',  # 论文
    '💻': '',  # 系统要求
    '⚠️': '',  # 警告
    '⚙️': '',  # 环境配置
    '🔍': '',  # 执行命令
    '✅': '',  # 成功
    '🔧': '',  # 修复
    '💡': '',  # 提示
    '📌': '',  # 方案
    '📦': '',  # 安装
    '📋': '',  # 表格
    '🎉': '',  # 完成
    '💾': '',  # 数据准备
    '📁': '',  # 目录
    '🗂️': '',  # 结构
    '🎯': '',  # 方案
    '🔧': '',  # 修复
    '❌': '',  # 错误
    '🚀': '',  # 训练
    '📊': '',  # 参数
    '📈': '',  # 曲线
    '🏆': '',  # 最佳
    '💾': '',  # 保存
    '⚠️': '',  # 警告
    '🎊': '',  # 恭喜
    '🕐': '',  # 时间
    '❓': '',  # 问题
    '🔄': '',  # 其他
    '🔬': '',  # 进阶
    '📄': '',  # 论文
}

# 替换所有emoji
for emoji, replacement in emoji_map.items():
    content = content.replace(emoji, replacement)

# 移除span class="icon"中的emoji
content = re.sub(r'<span class="icon">[^<]*</span>', '', content)

# 保存文件
with open('DeepHS-Net-env.html', 'w', encoding='utf-8') as f:
    f.write(content)

print("Emoji removed successfully!")
