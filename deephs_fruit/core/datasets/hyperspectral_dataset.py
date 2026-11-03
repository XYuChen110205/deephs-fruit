"""
================================================================================
Hyperspectral Dataset Module / 高光谱数据集模块
================================================================================
This module handles loading and preprocessing of hyperspectral fruit data.
/ 这个模块处理高光谱水果数据的加载和预处理。

PURPOSE / 用途:
    - Load hyperspectral images (.hdr + .bin files) / 加载高光谱图像
    - Cache preprocessed data for faster loading / 缓存预处理数据加速加载
    - Balance classes (if needed) / 平衡类别（如需要）
    - Handle data splits (train/val/test) / 处理数据划分

KEY CONCEPTS / 关键概念:
    - Hyperspectral Cube: 3D array (height x width x bands)
      高光谱立方体：3D数组（高×宽×波段）
    - Each pixel contains a full spectrum (224-252 wavelengths)
      每个像素包含完整光谱（224-252个波长）
    - Bands = color channels (but many more than RGB)
      波段 = 颜色通道（但比RGB多很多）
================================================================================
"""

from torch.utils.data import Dataset
import torch
import os
import hashlib
import time

from core.name_convention import *
from core.measurements import *
import core.util as util
import core.fruit_list as fruit_list
import cv2
import numpy as np
import copy


# ==============================================================================
# HyperspectralDataset Class / 高光谱数据集类
# ==============================================================================

class HyperspectralDataset(Dataset):
    """
    PyTorch Dataset for hyperspectral fruit images.
    / 高光谱水果图像的PyTorch数据集。
    
    This class:
    / 这个类：
    1. Loads hyperspectral data from disk / 从磁盘加载高光谱数据
    2. Caches preprocessed data / 缓存预处理数据
    3. Returns (image, label, wavelengths) tuples / 返回（图像，标签，波长）元组
    """
    
    def __init__(self, classification_type, records, data_path, balance=False, transform=None, input_size=(62, 62)):
        """
        Initialize dataset.
        / 初始化数据集。
        
        Args:
            classification_type: RIPENESS, FIRMNESS, or SUGAR / 分类类型
            records: List of fruit records / 水果记录列表
            data_path: Root directory of dataset / 数据集根目录
            balance: Whether to balance class sizes / 是否平衡类别大小
            transform: Data augmentation transforms / 数据增强变换
            input_size: Target image size (height, width) / 目标图像大小
        """
        assert len(records) > 0

        self.classification_type = classification_type
        self._org_records = records
        self.records = self._org_records
        self.data_path = data_path
        self.balance = balance
        self.input_size = input_size

        self.transform = transform

        # Preload all data into memory / 预加载所有数据到内存
        self._preload_data()

        # Balance classes if requested / 如需要则平衡类别
        if self.balance:
            self.records = self._org_records.copy()
            self._balance_classes()

    def __len__(self):
        """Return number of samples / 返回样本数量"""
        return len(self.records)

    def _cache_root(self):
        """
        Get cache directory path.
        / 获取缓存目录路径。
        """
        w, h = self.input_size
        # Bump version when prepare_fruit sanitization changes so stale .npy are not reused.
        # 当prepare_fruit清洗逻辑改变时提升版本号，避免重用旧的.npy文件
        return os.path.join(self.data_path, "_preload_cache", f"v2_{w}x{h}")

    def _cache_file_for(self, record):
        """
        Generate cache filename for a record.
        / 为记录生成缓存文件名。
        """
        rel_path = record.get_file_path()
        key = f"{rel_path}|{self.input_size[0]}x{self.input_size[1]}"
        digest = hashlib.md5(key.encode("utf-8")).hexdigest()
        return os.path.join(self._cache_root(), f"{digest}.npy")

    def _preload_data(self):
        """
        Preload all data into memory with caching.
        / 预加载所有数据到内存，使用缓存。
        
        First time: Load from raw files, process, save to cache
        / 第一次：从原始文件加载，处理，保存到缓存
        Subsequent times: Load from cache (much faster)
        / 后续：从缓存加载（快得多）
        """
        total = len(self.records)
        print(f"# Preload data: {total} records to load")
        start_time = time.time()
        cache_root = self._cache_root()
        os.makedirs(cache_root, exist_ok=True)
        print(f"# Cache dir: {cache_root}")

        self.fruit_data = {}
        cache_hits = 0
        for i, r in enumerate(self.records):
            t0 = time.time()
            cache_file = self._cache_file_for(r)
            source = "cache"
            if os.path.exists(cache_file):
                try:
                    data = np.load(cache_file, allow_pickle=False)
                except Exception:
                    data = prepare_fruit(r, self.data_path, input_size=self.input_size)
                    np.save(cache_file, data)
                    source = "rebuild"
                else:
                    cache_hits += 1
            else:
                data = prepare_fruit(r, self.data_path, input_size=self.input_size)
                np.save(cache_file, data)
                source = "build"

            self.fruit_data[r] = data
            elapsed = time.time() - t0

            total_elapsed = time.time() - start_time
            avg_per_record = total_elapsed / (i + 1)
            remaining = avg_per_record * (total - i - 1)

            print(f"  [{i+1}/{total}] {r.get_file_path()} "
                  f"| src: {source} "
                  f"| this: {elapsed:.1f}s "
                  f"| elapsed: {total_elapsed:.0f}s "
                  f"| ETA: {remaining:.0f}s")

        total_time = time.time() - start_time
        print(f"# done. Total preload time: {total_time:.1f}s | cache hits: {cache_hits}/{total}")

    def _balance_classes(self):
        """
        Balance class sizes by duplicating minority class samples.
        / 通过复制少数类样本来平衡类别大小。
        
        This prevents the model from being biased toward majority classes.
        / 这防止模型偏向多数类。
        """
        if self.classification_type == ClassificationType.RIPENESS:
            states = [RipenessState.UNRIPE, RipenessState.RIPE, RipenessState.PERFECT, RipenessState.NEAR_OVERRIPE,
                      RipenessState.OVERRIPE]
        if self.classification_type == ClassificationType.FIRMNESS:
            states = [FirmnessLevel.TOO_SOFT,
                      FirmnessLevel.READY, FirmnessLevel.TOO_HARD]
        if self.classification_type == ClassificationType.SUGAR:
            states = [SugarLevel.NOT_SWEET,
                      SugarLevel.READY, SugarLevel.TOO_SWEET]

        def get_label(record):
            if self.classification_type == ClassificationType.RIPENESS:
                return record.label.ripeness_state
            if self.classification_type == ClassificationType.FIRMNESS:
                return record.label.get_firmness_level()
            if self.classification_type == ClassificationType.SUGAR:
                return record.label.get_sugar_level()

        max_count = 0
        for s in states:
            count = len([r for r in self.records if get_label(r) == s])
            print("%s #: %i" % (s, count))
            max_count = max(max_count, count)

        self.balance_to = max_count

        print("# Augment data to get balanced classes of size %i" %
              self.balance_to)

        target_class_size = self.balance_to
        # the sets are unbalanced, so augment the data to get balance sets
        # 数据集不平衡，所以增强数据以获得平衡集
        for s in states:
            class_records = [r for r in self.records if get_label(r) == s]

            if len(class_records) == 0:
                continue

            print("# Augment: %s to %i elements" % (s, target_class_size))

            missing_objects_count = target_class_size - len(class_records)
            for i in range(missing_objects_count):
                new_record = class_records[np.random.randint(
                    0, len(class_records))]
                self.records = np.concatenate((self.records, [new_record]))

        print("# Data augmented")

        for s in states:
            count = len([r for r in self.records if get_label(r) == s])
            print("%s #: %i" % (s, count))

    def __getitem__(self, index):
        """
        Get a single sample.
        / 获取单个样本。
        
        Returns:
            (image, label, wavelengths) tuple
            （图像，标签，波长）元组
        """
        item = self.records[index]

        label = None

        # Convert label to integer / 将标签转换为整数
        if self.classification_type == ClassificationType.RIPENESS:
            label = torch.tensor(ripeness2int(item.label.ripeness_state))
        if self.classification_type == ClassificationType.FIRMNESS:
            label = torch.tensor(firmness2int(item.label.get_firmness_level()))
        if self.classification_type == ClassificationType.SUGAR:
            label = torch.tensor(sugar2int(item.label.get_sugar_level()))

        assert label is not None

        # Get wavelength values for this camera type / 获取此相机类型的波长值
        channel_wavelengths = torch.tensor(util.get_wavelengths_for(item.camera_type),
                                           requires_grad=False)

        item = torch.tensor(self.fruit_data[item])

        # Apply transforms (augmentation, normalization) / 应用变换（增强、归一化）
        if self.transform is not None:
            item, label, channel_wavelengths = self.transform([item, label,
                                                               channel_wavelengths])

        return item, label, channel_wavelengths


def collater(batch):
    """Custom collate function / 自定义合并函数"""
    x = [b[0] for b in batch]
    y = [b[1] for b in batch]

    return x, y


# ==============================================================================
# Label Merging Functions / 标签合并函数
# ==============================================================================

def merge_ripeness_levels(records):
    """
    Merge 5 ripeness classes into 3 classes.
    / 将5个成熟度类别合并为3个。
    
    Original 5 classes: unripe, ripe, perfect, near_overripe, overripe
    / 原始5类：未成熟、成熟、完美、近过熟、过熟
    
    Merged 3 classes:
    / 合并后3类：
    - unripe (未成熟)
    - perfect (成熟+完美)
    - overripe (近过熟+过熟)
    """

    for r in records:
        if r.label.ripeness_state == RipenessState.NEAR_OVERRIPE:
            r.label.ripeness_state = RipenessState.OVERRIPE
        elif r.label.ripeness_state == RipenessState.RIPE:
            r.label.ripeness_state = RipenessState.PERFECT

    return records


def extend_unripe_recordings(records):
    """
    Extend unripe labels to earlier recordings of same fruit.
    / 将未成熟标签扩展到同一水果的早期记录。
    
    Assumption: If a fruit is unripe on day 5, it was also unripe on day 1-4.
    / 假设：如果一个水果第5天是未成熟，那么第1-4天也是未成熟。
    """

    # find unripe fruit / 找到未成熟水果
    unripe_fruit = [r for r in records if r.is_labeled()
                        and (r.label.ripeness_state in
                             [RipenessState.UNRIPE, RipenessState.RIPE])]
    unripe_fruit_ids = [r.id for r in unripe_fruit]

    unripe_label = copy.copy(unripe_fruit[0].label)
    unripe_label.comment = "Extended by time assumption"

    for r in records:
        if not r.is_labeled() and r.id in unripe_fruit_ids:
            r.label = unripe_label

    return records


def extend_too_firm_recordings(records: [FruitRecord]):
    """
    Extend too-firm labels to earlier recordings.
    / 将过硬标签扩展到早期记录。
    """

    toofirm_fruit = [r for r in records if r.is_labeled() and r.label.firmness is not None
                        and (r.label.get_firmness_level()
                             in
                             [FirmnessLevel.TOO_HARD])]
    toofirm_fruit_ids = [r.id for r in toofirm_fruit]


    toofirm_label = copy.copy(toofirm_fruit[0].label)
    toofirm_label.comment = "Extended by time assumption"
    

    for r in records:
        if not r.is_labeled() and r.id in toofirm_fruit_ids:
            r.label = toofirm_label

    return records

def extend_too_unsweet_recordings(records: [FruitRecord]):
    """
    Extend too-unsweet labels to earlier recordings.
    / 将不够甜标签扩展到早期记录。
    """

    toounsweet_fruit = [r for r in records if r.is_labeled()
                        and r.label.sugar_content is not None and
                        (r.label.get_sugar_level() in [SugarLevel.NOT_SWEET])]
    toounsweet_fruit_ids = [r.id for r in toounsweet_fruit]


    toounsweet_label = copy.copy(toounsweet_fruit[0].label)
    toounsweet_label.comment = "Extended by time assumption"
    

    for r in records:
        if not r.is_labeled() and r.id in toounsweet_fruit_ids:
            r.label = toounsweet_label

    return records


# ==============================================================================
# Data Splitting Function / 数据划分函数
# ==============================================================================

def _get_ripeness_label(record):
    if record.label is None:
        return None
    if record.label.ripeness_state in [RipenessState.UNRIPE]:
        return 0
    if record.label.ripeness_state in [RipenessState.PERFECT]:
        return 1
    if record.label.ripeness_state in [RipenessState.OVERRIPE, RipenessState.NEAR_OVERRIPE]:
        return 2
    return None


def random_split_by_id(records, val_ratio=0.15, test_ratio=0.15, seed=42):
    """
    Split records by fruit ID with stratified class distribution.
    / 按水果ID分层划分，确保各类别分布一致。
    
    Key principle: Same fruit ID stays in ONE split only.
    / 关键原则：同一水果ID只属于一个划分集。
    
    This prevents information leakage where the model sees
    different ripeness stages of the same fruit across splits.
    / 这防止了模型在不同划分集中看到同一水果的不同成熟阶段。
    """
    import random as _random
    _random.seed(seed)

    # Group records by fruit ID / 按水果ID分组
    id_to_records = {}
    for r in records:
        key = r.id
        if key not in id_to_records:
            id_to_records[key] = []
        id_to_records[key].append(r)

    # Get unique fruit IDs and their majority class / 获取唯一水果ID及其主要类别
    id_list = list(id_to_records.keys())
    _random.shuffle(id_list)

    # Assign IDs to splits while balancing classes / 平衡各类别地分配ID到各划分集
    train_ids, val_ids, test_ids = [], [], []
    train_counts, val_counts, test_counts = {0: 0, 1: 0, 2: 0}, {0: 0, 1: 0, 2: 0}, {0: 0, 1: 0, 2: 0}

    total = len(records)
    target_val = int(total * val_ratio)
    target_test = int(total * test_ratio)
    target_train = total - target_val - target_test

    for fid in id_list:
        group = id_to_records[fid]
        # Majority class in this group / 该组的主要类别
        cls_counts = {0: 0, 1: 0, 2: 0}
        for r in group:
            c = _get_ripeness_label(r)
            if c is not None:
                cls_counts[c] = cls_counts.get(c, 0) + 1
        majority_cls = max(cls_counts, key=cls_counts.get)

        # Assign to the split that needs this class most / 分配到最需要该类别的划分集
        current_train = sum(train_counts.values())
        current_val = sum(val_counts.values())
        current_test = sum(test_counts.values())

        if current_val < target_val:
            assigned = 'val'
        elif current_test < target_test:
            assigned = 'test'
        else:
            assigned = 'train'

        if assigned == 'train':
            train_ids.append(fid)
            for c in cls_counts:
                train_counts[c] = train_counts.get(c, 0) + cls_counts[c]
        elif assigned == 'val':
            val_ids.append(fid)
            for c in cls_counts:
                val_counts[c] = val_counts.get(c, 0) + cls_counts[c]
        else:
            test_ids.append(fid)
            for c in cls_counts:
                test_counts[c] = test_counts.get(c, 0) + cls_counts[c]

    # Build final record lists / 构建最终记录列表
    train_records = []
    for fid in train_ids:
        train_records.extend(id_to_records[fid])
    val_records = []
    for fid in val_ids:
        val_records.extend(id_to_records[fid])
    test_records = []
    for fid in test_ids:
        test_records.extend(id_to_records[fid])

    print(f"Random split by fruit ID (seed={seed}):")
    print(f"  Train: {len(train_records)} records, {len(train_ids)} unique IDs")
    print(f"  Val:   {len(val_records)} records, {len(val_ids)} unique IDs")
    print(f"  Test:  {len(test_records)} records, {len(test_ids)} unique IDs")
    for name, recs in [("Train", train_records), ("Val", val_records), ("Test", test_records)]:
        cnt = {0: 0, 1: 0, 2: 0}
        for r in recs:
            c = _get_ripeness_label(r)
            if c is not None:
                cnt[c] = cnt.get(c, 0) + 1
        total_c = sum(cnt.values())
        if total_c > 0:
            print(f"    {name}: unripe={cnt[0]}({cnt[0]/total_c*100:.0f}%) "
                  f"perfect={cnt[1]}({cnt[1]/total_c*100:.0f}%) "
                  f"overripe={cnt[2]}({cnt[2]/total_c*100:.0f}%)")

    return np.array(train_records), np.array(val_records), np.array(test_records)


def get_records(fruit, camera_type, classification_type,
                extend_by_time_assumption=False, use_inter_ripeness_levels=False, allow_all_fruit_types=True,
                use_new_recordings=True,
                random_split=False, val_ratio=0.15, test_ratio=0.15, seed=42
               ):
    """
    Get train/val/test records for training.
    / 获取训练用的训练/验证/测试记录。
    
    This function:
    / 这个函数：
    1. Loads all records for specified fruit and camera / 加载指定水果和相机的所有记录
    2. Extends labels by time assumption (if enabled) / 按时间假设扩展标签
    3. Merges ripeness levels (if enabled) / 合并成熟度级别
    4. Splits into train/val/test / 划分为训练/验证/测试
       - Original: uses hardcoded test_set_fruits/val_set_fruits (val set too small!)
       - Random:   splits by fruit ID with balanced classes (recommended!)
    """

    full_set = all_fruits_new if use_new_recordings else all_fruits

    if fruit == Fruit.ALL and not allow_all_fruit_types:
        raise NotImplementedError("ALL fruit types is not supported for this classifier!")

    records = fruit_list.get_for_camera_type(fruit_list.get_for_fruit(
        full_set, fruit), camera_type)

    assert len(records) > 0

    # Extend labels based on time assumption / 基于时间假设扩展标签
    if extend_by_time_assumption:
        if classification_type == ClassificationType.RIPENESS:
            records = extend_unripe_recordings(records)
        elif classification_type == ClassificationType.FIRMNESS:
            records = extend_too_firm_recordings(records)
        elif classification_type == ClassificationType.SUGAR:
            records = extend_too_unsweet_recordings(records)

    records = fruit_list.get_labeled_fruits(records)

    # Merge 5 classes into 3 / 将5类合并为3类
    if use_inter_ripeness_levels:
        records = merge_ripeness_levels(records)

    # Filter by classification type / 按分类类型过滤
    if classification_type == ClassificationType.RIPENESS:
        records = np.concatenate((fruit_list.get_fruits_with_state(records, RipenessState.UNRIPE),
                                  fruit_list.get_fruits_with_state(
                                      records, RipenessState.PERFECT),
                                  fruit_list.get_fruits_with_state(records, RipenessState.OVERRIPE)))
    if classification_type == ClassificationType.FIRMNESS:
        records = np.concatenate((fruit_list.get_fruits_with_firmness_level(records, FirmnessLevel.TOO_SOFT),
                                  fruit_list.get_fruits_with_firmness_level(
                                      records, FirmnessLevel.READY),
                                  fruit_list.get_fruits_with_firmness_level(records, FirmnessLevel.TOO_HARD)))
    if classification_type == ClassificationType.SUGAR:
        records = np.concatenate((fruit_list.get_fruits_with_sugar_level(records, SugarLevel.NOT_SWEET),
                                  fruit_list.get_fruits_with_sugar_level(
                                      records, SugarLevel.READY),
                                  fruit_list.get_fruits_with_sugar_level(records, SugarLevel.TOO_SWEET)))

    if random_split:
        # Use new random split by fruit ID (recommended) / 使用新的按ID随机划分（推荐）
        train_records, val_records, test_records = random_split_by_id(
            records, val_ratio=val_ratio, test_ratio=test_ratio, seed=seed)
    else:
        # Use original hardcoded split / 使用原始硬编码划分
        rest_records, test_records = fruit_list.extract_test_data(records)
        train_records, val_records = fruit_list.extract_val_data(rest_records)

    return train_records, val_records, test_records


# ==============================================================================
# Data Preprocessing Functions / 数据预处理函数
# ==============================================================================

def add_border(_obj):
    """Add border to image / 为图像添加边框"""
    enlarged = np.zeros(
        (_obj.shape[0] + 2, _obj.shape[1] + 2, _obj.shape[2]), dtype=np.float32)
    enlarged[1:-1, 1:-1] = _obj
    return enlarged


def prepare_fruit(_f: FruitRecord, data_path: str, input_size):
    """
    Load and preprocess a single fruit hyperspectral image.
    / 加载并预处理单个水果高光谱图像。
    
    Steps:
    / 步骤：
    1. Load ENVI format data (.hdr + .bin) / 加载ENVI格式数据
    2. Resize to target size / 调整大小
    3. Clean NaN/Inf values / 清洗NaN/Inf值
    4. Clip to valid range / 裁剪到有效范围
    5. Reorder dimensions (bands first) / 重排维度（波段优先）
    """
    _header, _data = _f.load(data_path, is_already_referenced=True)
    _data = cv2.resize(_data, dsize=input_size, interpolation=cv2.INTER_CUBIC)
    _data = np.asarray(_data, dtype=np.float32)
    # Reflectance / ENVI can yield NaN or Inf.
    # 反射率/ENVI可能产生NaN或Inf
    _data = np.nan_to_num(_data, nan=0.0, posinf=1.0, neginf=0.0)
    np.copyto(_data, 0.0, where=~np.isfinite(_data))
    np.clip(_data, 0.0, 10.0, out=_data)
#    _data = add_border(_data)

    return bands_as_first_dimension(_data)

def bands_as_first_dimension(_obj):
    """
    Reorder dimensions from (H, W, B) to (B, H, W).
    / 将维度从（高，宽，波段）重排为（波段，高，宽）。
    
    PyTorch expects (channels, height, width) format.
    PyTorch需要（通道，高，宽）格式。
    """
    if isinstance(_obj, torch.Tensor):
        return _obj.permute(2, 0, 1)
    else:
        return _obj.transpose((2, 0, 1))


def bands_as_first_dimension_rev(_obj):
    """
    Reverse: from (B, H, W) to (H, W, B).
    / 反向：从（波段，高，宽）到（高，宽，波段）。
    """
    if isinstance(_obj, torch.Tensor):
        return _obj.permute(1, 2, 0)
    else:
        return _obj.transpose((1, 2, 0))
