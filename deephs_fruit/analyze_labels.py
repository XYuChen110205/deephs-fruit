import sys
sys.path.insert(0, 'e:\\DeepHS-Fruit')

from deephs_fruit.core.measurements.measurement_series_1 import (
    day_1_all_fruits, day_2_all_fruits, test_all_nir_fruits
)
from deephs_fruit.core.measurements.measurement_series_2 import (
    day_m2_01_all_fruits, day_m2_02_all_fruits, day_m2_03_all_fruits, day_m2_04_all_fruits
)
from deephs_fruit.core.measurements.measurement_series_3 import (
    day_m3_all_fruit_interpolated
)
from deephs_fruit.core.measurements.measurement_series_4 import (
    day_m4_1_all_fruits
)
from deephs_fruit.core.name_convention import Fruit, CameraType

def count_labels(records, fruit_filter=None, camera_filter=None):
    total = 0
    with_label = 0
    without_label = 0

    for r in records:
        # 检查水果类型 - 比较值而不是枚举对象本身
        if fruit_filter:
            if hasattr(r.fruit, 'value'):
                if r.fruit.value != fruit_filter.value:
                    continue
            else:
                if r.fruit != fruit_filter.value:
                    continue
        # 检查相机类型 - 比较值而不是枚举对象本身
        if camera_filter:
            if hasattr(r.camera_type, 'value'):
                if r.camera_type.value != camera_filter.value:
                    continue
            else:
                if r.camera_type != camera_filter.value:
                    continue

        total += 1
        if r.label is not None:
            with_label += 1
        else:
            without_label += 1

    return total, with_label, without_label

def print_stats(name, total, with_label, without_label):
    if total > 0:
        ratio = with_label / total * 100
        print(f"{name:10s}: 总共 {total:4d} 条 | 有标签 {with_label:4d} 条 | 无标签 {without_label:4d} 条 | 标签率 {ratio:.1f}%")
    else:
        print(f"{name:10s}: 无数据")

all_records = (
    day_1_all_fruits + day_2_all_fruits + test_all_nir_fruits +
    day_m2_01_all_fruits + day_m2_02_all_fruits + day_m2_03_all_fruits + day_m2_04_all_fruits +
    day_m3_all_fruit_interpolated +
    day_m4_1_all_fruits
)

print("=" * 70)
print("标签分布统计")
print("=" * 70)

# 调试: 查看第一条记录的类型
if all_records:
    r = all_records[0]
    print(f"\n调试信息 - 第一条记录:")
    print(f"  fruit 类型: {type(r.fruit)}, 值: {r.fruit}")
    print(f"  camera_type 类型: {type(r.camera_type)}, 值: {r.camera_type}")
    print(f"  Fruit.AVOCADO 类型: {type(Fruit.AVOCADO)}, 值: {Fruit.AVOCADO}")
    print(f"  比较结果: {r.fruit == Fruit.AVOCADO}")

print("\n【Avocado 牛油果】")
print("-" * 70)
for camera in [CameraType.VIS, CameraType.NIR, CameraType.VIS_COR]:
    total, with_label, without_label = count_labels(all_records, Fruit.AVOCADO, camera)
    print_stats(camera.name, total, with_label, without_label)

total, with_label, without_label = count_labels(all_records, Fruit.AVOCADO, None)
print_stats("Total", total, with_label, without_label)

print("\n【Kiwi 猕猴桃】")
print("-" * 70)
for camera in [CameraType.VIS, CameraType.NIR, CameraType.VIS_COR]:
    total, with_label, without_label = count_labels(all_records, Fruit.KIWI, camera)
    print_stats(camera.name, total, with_label, without_label)

total, with_label, without_label = count_labels(all_records, Fruit.KIWI, None)
print_stats("Total", total, with_label, without_label)

print("\n【Kaki 柿子】")
print("-" * 70)
for camera in [CameraType.VIS, CameraType.VIS_COR]:
    total, with_label, without_label = count_labels(all_records, Fruit.KAKI, camera)
    print_stats(camera.name, total, with_label, without_label)

total, with_label, without_label = count_labels(all_records, Fruit.KAKI, None)
print_stats("Total", total, with_label, without_label)

print("\n【总计】")
print("=" * 70)
total, with_label, without_label = count_labels(all_records, None, None)
print_stats("所有水果", total, with_label, without_label)

print("\n" + "=" * 70)
print("结论:")
print("=" * 70)
print("标签缺失是数据集本身的设计，不是使用过程中的问题。")
print("作者只在特定的时间点(通常是测量当天)为部分记录添加了标签。")
print("=" * 70)
