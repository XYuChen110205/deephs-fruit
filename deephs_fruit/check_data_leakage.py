import os, sys
sys.path.insert(0, 'E:\\DeepHS-Fruit\\deephs_fruit')

from core.name_convention import *
from core.measurements.val_set_fruits import val_set_fruits
from core.measurements.test_set_fruits import test_set_fruits
from core.measurements import all_fruits_new
import core.fruit_list as fruit_list

from core.datasets.hyperspectral_dataset import get_records, merge_ripeness_levels, extend_unripe_recordings

train_records, val_records, test_records = get_records(
    Fruit.AVOCADO,
    CameraType.VIS,
    ClassificationType.RIPENESS,
    use_inter_ripeness_levels=True,
    extend_by_time_assumption=True,
    allow_all_fruit_types=True
)

print("="*60)
print("DATA LEAKAGE CHECK / 数据泄露检查")
print("="*60)

print(f"\nTrain set size: {len(train_records)}")
print(f"Val set size: {len(val_records)}")
print(f"Test set size: {len(test_records)}")

# Check if same fruit ID appears in both train and test
train_ids = set(r.id for r in train_records)
val_ids = set(r.id for r in val_records)
test_ids = set(r.id for r in test_records)

print("\n--- Same ID in Train and Val ---")
train_val_shared = train_ids & val_ids
if train_val_shared:
    print(f"  [LEAK] {len(train_val_shared)} IDs shared!")
    for shared_id in sorted(train_val_shared, key=lambda x: x.value):
        train_recs = [r for r in train_records if r.id == shared_id]
        val_recs = [r for r in val_records if r.id == shared_id]
        print(f"    ID {shared_id.value}:")
        print(f"      Train: {[(r.day.value, r.side.value) for r in train_recs]}")
        print(f"      Val:   {[(r.day.value, r.side.value) for r in val_recs]}")
else:
    print("  [OK] No leakage")

print("\n--- Same ID in Train and Test ---")
train_test_shared = train_ids & test_ids
if train_test_shared:
    print(f"  [LEAK] {len(train_test_shared)} IDs shared!")
    for shared_id in sorted(train_test_shared, key=lambda x: x.value):
        train_recs = [r for r in train_records if r.id == shared_id]
        test_recs = [r for r in test_records if r.id == shared_id]
        print(f"    ID {shared_id.value}:")
        print(f"      Train: {[(r.day.value, r.side.value) for r in train_recs]}")
        print(f"      Test:  {[(r.day.value, r.side.value) for r in test_recs]}")
else:
    print("  [OK] No leakage")

print("\n--- Same ID in Val and Test ---")
val_test_shared = val_ids & test_ids
if val_test_shared:
    print(f"  [LEAK] {len(val_test_shared)} IDs shared!")
    for shared_id in sorted(val_test_shared, key=lambda x: x.value):
        val_recs_val = [r for r in val_records if r.id == shared_id]
        test_recs_val = [r for r in test_records if r.id == shared_id]
        print(f"    ID {shared_id.value}:")
        print(f"      Val:  {[(r.day.value, r.side.value) for r in val_recs_val]}")
        print(f"      Test: {[(r.day.value, r.side.value) for r in test_recs_val]}")
else:
    print("  [OK] No leakage")

# Check class distribution
print("\n" + "="*60)
print("CLASS DISTRIBUTION / 类别分布")
print("="*60)

from core.measurements import RipenessState

def r2int(state):
    mapping = {
        RipenessState.UNRIPE: 0,
        RipenessState.PERFECT: 1,
        RipenessState.OVERRIPE: 2,
    }
    return mapping.get(state, -1)

for name, records in [("Train", train_records), ("Val", val_records), ("Test", test_records)]:
    counts = {0: 0, 1: 0, 2: 0}
    for r in records:
        s = r2int(r.label.ripeness_state)
        counts[s] = counts.get(s, 0) + 1
    total = sum(counts.values())
    if total == 0: continue
    print(f"\n{name}: {total} samples")
    print(f"  Unripe(0):   {counts[0]} ({counts[0]/total*100:.1f}%)")
    print(f"  Perfect(1):  {counts[1]} ({counts[1]/total*100:.1f}%)")
    print(f"  Overripe(2): {counts[2]} ({counts[2]/total*100:.1f}%)")

print("\n" + "="*60)
print("ALL UNIQUE IDs IN TRAIN")
print("="*60)
print(sorted(train_ids, key=lambda x: x.value))
print("\nALL UNIQUE IDs IN VAL")
print(sorted(val_ids, key=lambda x: x.value))
print("\nALL UNIQUE IDs IN TEST")
print(sorted(test_ids, key=lambda x: x.value))
