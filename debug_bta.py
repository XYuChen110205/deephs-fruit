"""调试 BTA 为什么没扩展"""
import sys; sys.path.insert(0, r'e:\DeepHS-Fruit\deephs_fruit')
from core.datasets.hyperspectral_dataset import get_records
from core.name_convention import *
from collections import defaultdict

# 1. 看看 get_records 返回的记录是否已经过滤了无标注的
train, val, test = get_records(Fruit.AVOCADO, CameraType.VIS, ClassificationType.RIPENESS,
    use_inter_ripeness_levels=True, random_split=True, seed=23312323)

print(f'train_records: {len(train)}')
print(f'有标注: {sum(1 for r in train if r.is_labeled())}')
print(f'无标注: {sum(1 for r in train if not r.is_labeled())}')

# 2. 获取完整数据集（包含无标注的）
from core.measurements import all_fruits_new
from core import fruit_list
full_set = fruit_list.get_for_camera_type(
    fruit_list.get_for_fruit(all_fruits_new, Fruit.AVOCADO), CameraType.VIS)
print(f'\n完整 avocado VIS 记录: {len(full_set)}')
print(f'有标注: {sum(1 for r in full_set if r.is_labeled())}')
print(f'无标注: {sum(1 for r in full_set if not r.is_labeled())}')

# 3. 按水果ID分组看看
id_to_recs = defaultdict(list)
for r in full_set:
    id_to_recs[r.id].append(r)

print(f'\n水果ID数量: {len(id_to_recs)}')
for fid, recs in sorted(id_to_recs.items(), key=lambda x: str(x[0]))[:3]:
    labeled = [r for r in recs if r.is_labeled()]
    print(f'  ID {fid}: 共{len(recs)}条, 有标注{len(labeled)}条')
    for r in recs:
        if r.is_labeled():
            print(f'    {r.day.value} {r.side.value} -> {r.label.ripeness_state}')
        else:
            print(f'    {r.day.value} {r.side.value} -> 无标注')
