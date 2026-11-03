import os
import glob
import re

# 找到最新的wandb运行目录
wandb_dir = r'E:\DeepHS-Fruit\deephs_fruit\wandb'
run_dirs = glob.glob(os.path.join(wandb_dir, 'offline-run-*'))
run_dirs.sort(key=os.path.getmtime)
latest_run_dir = run_dirs[-1]

print(f"最新运行目录: {os.path.basename(latest_run_dir)}")
print("="*60)

# 读取wandb文件
run_id = os.path.basename(latest_run_dir).split('-')[-1]
wandb_file = os.path.join(latest_run_dir, f'run-{run_id}.wandb')

if os.path.exists(wandb_file):
    # 直接读取二进制文件中的关键指标
    with open(wandb_file, 'rb') as f:
        data = f.read()
    
    # 从文件末尾开始搜索（最新的记录）
    tail = data[-100000:]
    
    # 显示训练配置信息
    print("\n训练配置:")
    
    # 搜索命令行参数
    config_match = re.search(rb'--num_epochs\s*\*?(\d+)', data)
    if config_match:
        print(f"  Epochs: {int(config_match.group(1))}")
    
    config_match = re.search(rb'\*--lr\*\*?(\d+\.\d+)', data)
    if config_match:
        print(f"  Learning Rate: {float(config_match.group(1))}")
    
    config_match = re.search(rb'\*--batch_size\*\*?(\d+)', data)
    if config_match:
        print(f"  Batch Size: {int(config_match.group(1))}")
        
    # 显示数据集划分
    print("\n数据集划分:")
    if b'Random split by fruit ID' in data:
        print("  使用随机ID划分 (random_split_by_id)")
    
    # 搜索训练/验证/测试数量
    train_match = re.search(rb'Train:\s*(\d+)\s*records', data)
    val_match = re.search(rb'Val:\s*(\d+)\s*records', data)
    test_match = re.search(rb'Test:\s*(\d+)\s*records', data)
    
    if train_match:
        print(f"  Train: {int(train_match.group(1))} records")
    if val_match:
        print(f"  Val: {int(val_match.group(1))} records")
    if test_match:
        print(f"  Test: {int(test_match.group(1))} records")
    
    # 查找class分布
    print("\n类别分布:")
    class_pattern = rb'(Train|Val|Test):\s*unripe=(\d+)\((\d+)%\)\s*perfect=(\d+)\((\d+)%\)\s*overripe=(\d+)\((\d+)%\)'
    matches = re.findall(class_pattern, data)
    for match in matches[-3:]:  # 最后3个（应该是train/val/test）
        split, u_cnt, u_pct, p_cnt, p_pct, o_cnt, o_pct = match
        print(f"  {split.decode()}: unripe={int(u_cnt)}({u_pct.decode()}%) perfect={int(p_cnt)}({p_pct.decode()}%) overripe={int(o_cnt)}({o_pct.decode()}%)")

print("\n" + "="*60)
