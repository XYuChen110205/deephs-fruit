"""
================================================================================
Name Convention Module / 命名约定模块
================================================================================
This module defines all the enums, classes, and naming conventions used
throughout the DeepHS-Fruit project.
/ 本模块定义了整个DeepHS-Fruit项目中使用的所有枚举、类和命名约定。

PURPOSE / 用途:
    - Standardize data identifiers (fruit type, camera, day, etc.)
      标准化数据标识符（水果类型、相机、天数等）
    - Define label classes for ripeness, firmness, and sugar content
      定义成熟度、硬度和糖度的标签类别
    - Provide conversion utilities between different representations
      提供不同表示之间的转换工具

KEY CONCEPTS / 关键概念:
    - FruitRecord: A single fruit measurement with metadata
      FruitRecord：带有元数据的单个水果测量
    - RipenessState: The ripeness classification labels
      RipenessState：成熟度分类标签
    - CameraType: VIS (visible), NIR (near-infrared), VIS_COR
      CameraType：VIS（可见光）、NIR（近红外）、VIS_COR
================================================================================
"""

import os
import enum
from typing import Union, Optional

import core.spectral_io as spectral_io


# ==============================================================================
# Enum Definitions / 枚举定义
# ==============================================================================

class Side(enum.Enum):
    """
    Side of the fruit being imaged.
    / 成像的水果侧面。
    
    FRONT: The main side facing camera / 朝向相机的主要侧面
    BACK: The opposite side / 相对的另一面
    """
    FRONT = 'front'
    BACK = 'back'


class Day(enum.Enum):
    """
    Measurement day identifiers.
    / 测量天数标识符。
    
    The dataset was collected over multiple measurement series (days).
    Each day represents a different batch of fruits at different ripening stages.
    / 数据集是在多个测量系列（天）中收集的。每天代表不同成熟阶段的不同批次水果。
    """
    TEST_1 = 'test'
    TEST_2 = 'test2'
    DAY_1 = 'day_01'
    DAY_2 = 'day_02'
    DAY_3 = 'day_03'
    DAY_4 = 'day_04'
    DAY_5 = 'day_05'
    DAY_6 = 'day_06'
    DAY_7 = 'day_07'
    DAY_8 = 'day_08'
    DAY_9 = 'day_09'
    DAY_10 = 'day_10'
    DAY_11 = 'day_11'
    DAY_M2_1 = 'day_m2_01'
    DAY_M2_2 = 'day_m2_02'
    DAY_M2_3 = 'day_m2_03'
    DAY_M2_4 = 'day_m2_04'
    DAY_M2_5 = 'day_m2_05'
    DAY_M2_6 = 'day_m2_06'
    DAY_M2_7 = 'day_m2_07'
    DAY_M2_8 = 'day_m2_08'
    DAY_M2_9 = 'day_m2_09'
    DAY_M2_10 = 'day_m2_10'
    DAY_M2_11 = 'day_m2_11'
    DAY_M2_12 = 'day_m2_12'
    DAY_M2_13 = 'day_m2_13'
    DAY_M2_14 = 'day_m2_14'
    DAY_M2_15 = 'day_m2_15'
    DAY_M2_16 = 'day_m2_16'
    DAY_M2_17 = 'day_m2_17'
    DAY_M3_1 = 'day_1_m3'
    DAY_M3_2 = 'day_2_m3'
    DAY_M3_3 = 'day_3_m3'
    DAY_M3_4 = 'day_4_m3'
    DAY_M3_5 = 'day_5_m3'
    DAY_M3_6 = 'day_6_m3'
    DAY_M3_7 = 'day_7_m3'
    DAY_M3_8 = 'day_8_m3'
    DAY_M3_9 = 'day_9_m3'
    DAY_M3_10 = 'day_10_m3'
    DAY_M3_11 = 'day_11_m3'
    DAY_M3_12 = 'day_12_m3'
    DAY_M4_1 = 'day_m4_01'
    DAY_M4_2 = 'day_m4_02'
    DAY_M4_3 = 'day_m4_03'
    DAY_M4_4 = 'day_m4_04'
    DAY_M4_5 = 'day_m4_05'
    DAY_M4_6 = 'day_m4_06'
    DAY_M4_7 = 'day_m4_07'
    DAY_M4_8 = 'day_m4_08'
    DAY_M4_9 = 'day_m4_09'


class CameraType(enum.Enum):
    """
    Hyperspectral camera types.
    / 高光谱相机类型。
    
    VIS: Visible light (400-1000nm), 224 bands
         可见光（400-1000nm），224个波段
    VIS_COR: Corning microHSI (408-902nm), 224 bands
             Corning microHSI（408-902nm），224个波段
    NIR: Near-infrared (900-1700nm), 252 bands
         近红外（900-1700nm），252个波段
    RGB: Regular color camera (not hyperspectral)
         普通彩色相机（非高光谱）
    """
    VIS = 'VIS'
    VIS_COR = 'VIS_COR'
    NIR = 'NIR'
    RGB = 'RGB'


class Fruit(enum.Enum):
    """
    Types of fruits in the dataset.
    / 数据集中的水果类型。
    
    AVOCADO: 牛油果 (主要研究对象)
    KIWI: 猕猴桃
    MANGO: 芒果
    KAKI: 柿子
    PAPAYA: 木瓜
    ALL: 所有水果（用于跨水果实验）
    """
    AVOCADO = 'Avocado'
    KIWI = 'Kiwi'
    MANGO = 'Mango'
    KAKI = 'Kaki'
    PAPAYA = 'Papaya'
    ALL = 'All'


class ID(enum.Enum):
    """
    Fruit identifier (unique within a day).
    / 水果标识符（每天内唯一）。
    
    Each fruit in a measurement batch gets a unique ID from 01-99.
    / 每个测量批次中的水果获得01-99的唯一ID。
    """
    UNKNOWN = '?'
    ID_1 = '01'
    ID_2 = '02'
    ID_3 = '03'
    ID_4 = '04'
    ID_5 = '05'
    ID_6 = '06'
    ID_7 = '07'
    ID_8 = '08'
    ID_9 = '09'
    ID_10 = '10'
    ID_11 = '11'
    ID_12 = '12'
    ID_13 = '13'
    ID_14 = '14'
    ID_15 = '15'
    ID_16 = '16'
    ID_17 = '17'
    ID_18 = '18'
    ID_19 = '19'
    ID_20 = '20'
    ID_21 = '21'
    ID_22 = '22'
    ID_23 = '23'
    ID_24 = '24'
    ID_25 = '25'
    ID_26 = '26'
    ID_27 = '27'
    ID_28 = '28'
    ID_29 = '29'
    ID_30 = '30'
    ID_31 = '31'
    ID_32 = '32'
    ID_33 = '33'
    ID_34 = '34'
    ID_35 = '35'
    ID_36 = '36'
    ID_37 = '37'
    ID_38 = '38'
    ID_39 = '39'
    ID_40 = '40'
    ID_41 = '41'
    ID_42 = '42'
    ID_43 = '43'
    ID_44 = '44'
    ID_45 = '45'
    ID_46 = '46'
    ID_47 = '47'
    ID_48 = '48'
    ID_49 = '49'
    ID_50 = '50'
    ID_51 = '51'
    ID_52 = '52'
    ID_53 = '53'
    ID_54 = '54'
    ID_55 = '55'
    ID_56 = '56'
    ID_57 = '57'
    ID_58 = '58'
    ID_59 = '59'
    ID_60 = '60'
    ID_61 = '61'
    ID_62 = '62'
    ID_63 = '63'
    ID_64 = '64'
    ID_65 = '65'
    ID_66 = '66'
    ID_67 = '67'
    ID_68 = '68'
    ID_69 = '69'
    ID_70 = '70'
    ID_71 = '71'
    ID_72 = '72'
    ID_73 = '73'
    ID_74 = '74'
    ID_75 = '75'
    ID_76 = '76'
    ID_77 = '77'
    ID_78 = '78'
    ID_79 = '79'
    ID_80 = '80'
    ID_81 = '81'
    ID_82 = '82'
    ID_83 = '83'
    ID_84 = '84'
    ID_85 = '85'
    ID_86 = '86'
    ID_87 = '87'
    ID_88 = '88'
    ID_89 = '89'
    ID_90 = '90'
    ID_91 = '91'
    ID_92 = '92'
    ID_93 = '93'
    ID_94 = '94'
    ID_95 = '95'
    ID_96 = '96'
    ID_97 = '97'
    ID_98 = '98'
    ID_99 = '99'


class DATASET_TYPE(enum.Enum):
    """
    Dataset split type.
    / 数据集划分类型。
    """
    TRAIN = "train",
    VAL = "validation",
    TEST = "test"


# ==============================================================================
# Ripeness State Enums / 成熟度状态枚举
# ==============================================================================

class RipenessState(enum.Enum):
    """
    Ripeness classification labels.
    / 成熟度分类标签。
    
    Original 5-class system:
    / 原始5类系统：
        UNRIPE: 未成熟 - Hard, green, not ready to eat
        RIPE: 成熟 - Edible but firm
        PERFECT: 完美 - Optimal eating quality
        NEAR_OVERRIPE: 近过熟 - Very soft, may be overripe
        OVERRIPE: 过熟 - Too soft, likely spoiled
    
    For classification, we typically merge to 3 classes:
    / 分类时通常合并为3类：
        unripe (UNRIPE)
        perfect (RIPE + PERFECT)
        overripe (NEAR_OVERRIPE + OVERRIPE)
    """
    UNRIPE = 'unripe'           # 未成熟
    RIPE = 'ripe'               # 成熟
    PERFECT = 'perfect'         # 完美
    NEAR_OVERRIPE = 'near_overripe'  # 近过熟
    OVERRIPE = 'overripe'       # 过熟
    UNKNOWN = 'unknown'         # 未知


def ripeness2int(_ripeness_state: RipenessState):
    """
    Convert RipenessState to integer.
    / 将RipenessState转换为整数。
    
    Mapping / 映射:
        UNRIPE -> 0
        PERFECT -> 1
        OVERRIPE -> 2
        RIPE -> 3 (mapped to perfect in 3-class system)
        NEAR_OVERRIPE -> 4 (mapped to overripe in 3-class system)
    """
    if _ripeness_state == RipenessState.UNRIPE:
        return 0
    if _ripeness_state == RipenessState.RIPE:
        return 3
    if _ripeness_state == RipenessState.PERFECT:
        return 1
    if _ripeness_state == RipenessState.NEAR_OVERRIPE:
        return 4
    if _ripeness_state == RipenessState.OVERRIPE:
        return 2


def int2ripeness(_ripeness_int: int):
    """
    Convert integer to RipenessState.
    / 将整数转换为RipenessState。
    """
    if _ripeness_int == 0:
        return RipenessState.UNRIPE
    if _ripeness_int == 3:
        return RipenessState.RIPE
    if _ripeness_int == 1:
        return RipenessState.PERFECT
    if _ripeness_int == 4:
        return RipenessState.NEAR_OVERRIPE
    if _ripeness_int == 2:
        return RipenessState.OVERRIPE


def ripeness2color(_ripeness: RipenessState):
    """
    Get color code for ripeness state (for plotting).
    / 获取成熟度状态的颜色代码（用于绘图）。
    
    Returns:
        'g' (green) for unripe / 未成熟-绿色
        'y' (yellow) for ripe/perfect / 成熟-黄色
        'r' (red) for overripe / 过熟-红色
        'b' (blue) for unknown / 未知-蓝色
    """
    if _ripeness == RipenessState.UNRIPE:
        return 'g'
    if _ripeness == RipenessState.RIPE or _ripeness == RipenessState.PERFECT:
        return 'y'
    if _ripeness == RipenessState.OVERRIPE:
        return 'r'
    return 'b'


# ==============================================================================
# Firmness Level Enums / 硬度级别枚举
# ==============================================================================

class FirmnessLevel(enum.Enum):
    """
    Firmness classification labels.
    / 硬度分类标签。
    
    Based on penetrometer measurements (g/cm²).
    / 基于 penetrometer 测量值（g/cm²）。
    """
    TOO_HARD = 'too_hard'   # 太硬 - Not ready to eat
    READY = 'ready'         # 刚好 - Optimal firmness
    TOO_SOFT = 'too_soft'   # 太软 - May be overripe


def firmness2int(_ripeness_state: FirmnessLevel):
    """Convert FirmnessLevel to integer. / 将FirmnessLevel转换为整数。"""
    if _ripeness_state == FirmnessLevel.TOO_HARD:
        return 0
    if _ripeness_state == FirmnessLevel.READY:
        return 1
    if _ripeness_state == FirmnessLevel.TOO_SOFT:
        return 2


def int2firmness(_i: int):
    """Convert integer to FirmnessLevel. / 将整数转换为FirmnessLevel。"""
    if _i == 0:
        return FirmnessLevel.TOO_HARD
    if _i == 1:
        return FirmnessLevel.READY
    if _i == 2:
        return FirmnessLevel.TOO_SOFT


def firmness2color(_ripeness: FirmnessLevel):
    """Get color for firmness level. / 获取硬度级别的颜色。"""
    if _ripeness == FirmnessLevel.TOO_HARD:
        return 'g'
    if _ripeness == FirmnessLevel.READY:
        return 'y'
    if _ripeness == FirmnessLevel.TOO_SOFT:
        return 'r'
    return 'b'


# ==============================================================================
# Sugar Level Enums / 糖度级别枚举
# ==============================================================================

class SugarLevel(enum.Enum):
    """
    Sugar content classification labels.
    / 糖度分类标签。
    
    Based on Brix measurements (°Brix).
    / 基于 Brix 测量值（°Brix）。
    """
    NOT_SWEET = 'not_sweet'   # 不够甜
    READY = 'ready'           # 刚好
    TOO_SWEET = 'too_sweet'   # 太甜


def sugar2int(_ripeness_state: SugarLevel):
    """Convert SugarLevel to integer. / 将SugarLevel转换为整数。"""
    if _ripeness_state == SugarLevel.NOT_SWEET:
        return 0
    if _ripeness_state == SugarLevel.READY:
        return 1
    if _ripeness_state == SugarLevel.TOO_SWEET:
        return 2


def int2sugar(_i: int):
    """Convert integer to SugarLevel. / 将整数转换为SugarLevel。"""
    if _i == 0:
        return SugarLevel.NOT_SWEET
    if _i == 1:
        return SugarLevel.READY
    if _i == 2:
        return SugarLevel.TOO_SWEET


def sugar2color(_ripeness: SugarLevel):
    """Get color for sugar level. / 获取糖度级别的颜色。"""
    if _ripeness == SugarLevel.NOT_SWEET:
        return 'g'
    if _ripeness == SugarLevel.READY:
        return 'y'
    if _ripeness == SugarLevel.TOO_SWEET:
        return 'r'
    return 'b'


# ==============================================================================
# Path and Naming Functions / 路径和命名函数
# ==============================================================================

def get_file_path(fruit: Fruit, side: Side, day: Day, id: ID, camera_type: CameraType):
    """
    Generate file path for a fruit record.
    / 为水果记录生成文件路径。
    
    Format: Fruit/CAMERA/day/filename
    格式：水果/相机/日期/文件名
    
    Example / 示例:
        Avocado/VIS/day_01/avocado_day_01_01_front
    """
    name = get_name(fruit, id, side, day)
    file_name = os.path.join(fruit.value.capitalize(), camera_type.value.upper(), day.value.lower(), name)
    return file_name


def get_name(fruit: Fruit, id: ID, side: Side, day: Day):
    """
    Generate base filename (without camera type).
    / 生成基本文件名（不含相机类型）。
    """
    if day == Day.TEST_1 or day == Day.TEST_2:
        name = fruit.value.lower() + "_" + day.value.lower() + "_" + side.value.lower()
    else:
        name = fruit.value.lower() + "_" + day.value.lower() + "_" + \
            id.value + "_" + side.value.lower()
    return name


def get_unique_name(fruit: Fruit, id: ID, side: Side, day: Day, camera_type: CameraType, postfix: str = None):
    """
    Generate unique identifier including camera type.
    / 生成包含相机类型的唯一标识符。
    """
    if postfix is None:
        name = fruit.value.lower() + "_" + day.value.lower() + "_" + id.value + \
            "_" + side.value.lower() + "_" + camera_type.value.lower()
    else:
        name = fruit.value.lower() + "_" + day.value.lower() + "_" + id.value + \
            "_" + side.value.lower() + "_" + camera_type.value.lower() + "_" + postfix
    return name


# ==============================================================================
# Label Classes / 标签类
# ==============================================================================

class AvocadoLabel:
    """
    Label for avocado fruit.
    / 牛油果标签。
    
    Contains all measurements from destructive testing.
    / 包含破坏性测试的所有测量值。
    
    Attributes:
        init_weight: Initial weight in grams / 初始重量（克）
        end_weight: Final weight after storage / 储存后最终重量
        storage_days: Days since harvest / 储存天数
        firmness: Penetrometer reading (g/cm²) / 硬度计读数
        ripeness_state: Expert-assigned ripeness label / 专家分配的成熟度标签
        comment: Additional notes / 额外注释
    """
    
    def __init__(self, _init_weight: int, _end_weight: int, _storage_days: int, _firmness: int,
                 _ripeness_state: RipenessState, _comment: str = None):
        self.init_weight = _init_weight
        self.end_weight = _end_weight
        self.storage_days = _storage_days
        self.firmness = _firmness  # in g / cm^2
        self.ripeness_state = _ripeness_state
        self.comment = _comment

    def __str__(self):
        if self.comment is not None:
            return "{%s, firmness: %s, comment: '%s'}" % (
                self.ripeness_state.value, self.firmness, self.comment)
        else:
            return "{%s, firmness: %s}" % (
                self.ripeness_state.value, self.firmness)

    def get_firmness_level(self) -> FirmnessLevel:
        """
        Convert firmness value to categorical level.
        / 将硬度值转换为分类级别。
        
        Thresholds for avocado / 牛油果阈值：
            > 1200 g/cm²: TOO_HARD
            900-1200 g/cm²: READY
            < 900 g/cm²: TOO_SOFT
        """
        if self.firmness > 1200:
            return FirmnessLevel.TOO_HARD
        elif self.firmness < 900:
            return FirmnessLevel.TOO_SOFT
        else:
            return FirmnessLevel.READY


class SweetFruitLabel:
    """
    Label for fruits with sugar content (kiwi, mango, kaki, papaya).
    / 含糖分水果的标签（猕猴桃、芒果、柿子、木瓜）。
    """
    
    def __init__(self, _init_weight: int, _end_weight: int, _storage_days: int, _firmness: int, _sugar_content: float,
                 _ripeness_state: RipenessState, _comment: str = None):
        self.init_weight = _init_weight
        self.end_weight = _end_weight
        self.storage_days = _storage_days
        self.firmness = _firmness  # in g / cm^2
        self.sugar_content = _sugar_content  # in °Brix
        self.ripeness_state = _ripeness_state
        self.comment = _comment

    def __str__(self):
        if self.comment is not None:
            return "{%s, firmness: %s, sugar content: %s, comment: '%s'}" % (
                self.ripeness_state.value, self.firmness, self.sugar_content, self.comment)
        else:
            return "{%s, firmness: %s, sugar content: %s}" % (
                self.ripeness_state.value, self.firmness, self.sugar_content)

    def get_firmness_level(self) -> FirmnessLevel:
        """To be implemented by subclasses. / 由子类实现。"""
        raise NotImplementedError()
    
    def get_sugar_level(self) -> SugarLevel:
        """To be implemented by subclasses. / 由子类实现。"""
        raise NotImplementedError()


class KiwiLabel(SweetFruitLabel):
    """Label for kiwi fruit. / 猕猴桃标签。"""
    
    def __init__(self, _init_weight: int, _end_weight: int, _storage_days: int, _firmness: int, _sugar_content: float,
                 _ripeness_state: RipenessState, _comment: str = None):
        super().__init__(_init_weight, _end_weight, _storage_days, _firmness, _sugar_content, _ripeness_state,
                         _comment)

    def get_firmness_level(self) -> FirmnessLevel:
        """Kiwi firmness thresholds: >1500 (hard), 1000-1500 (ready), <1000 (soft)"""
        if self.firmness > 1500:
            return FirmnessLevel.TOO_HARD
        elif self.firmness < 1000:
            return FirmnessLevel.TOO_SOFT
        else:
            return FirmnessLevel.READY

    def get_sugar_level(self) -> SugarLevel:
        """Kiwi sugar thresholds: <15.5°Brix (not sweet), 15.5-17°Brix (ready), >17°Brix (too sweet)"""
        if self.sugar_content < 15.5:
            return SugarLevel.NOT_SWEET
        elif self.sugar_content > 17:
            return SugarLevel.TOO_SWEET
        else:
            return SugarLevel.READY


class KakiLabel(SweetFruitLabel):
    """Label for kaki (persimmon). / 柿子标签。"""
    
    def __init__(self, _init_weight: int, _end_weight: int, _storage_days: int, _firmness: int, _sugar_content: float,
                 _ripeness_state: RipenessState, _comment: str = None):
        super().__init__(_init_weight, _end_weight, _storage_days, _firmness, _sugar_content, _ripeness_state,
                         _comment)

    def get_sugar_level(self) -> SugarLevel:
        """Kaki sugar thresholds: <20°Brix (not sweet), 20-22°Brix (ready), >22°Brix (too sweet)"""
        if self.sugar_content < 20:
            return SugarLevel.NOT_SWEET
        elif self.sugar_content > 22:
            return SugarLevel.TOO_SWEET
        else:
            return SugarLevel.READY

    def get_firmness_level(self) -> FirmnessLevel:
        """Kaki firmness thresholds: >1500 (hard), 500-1500 (ready), <500 (soft)"""
        if self.firmness > 1500:
            return FirmnessLevel.TOO_HARD
        elif self.firmness < 500:
            return FirmnessLevel.TOO_SOFT
        else:
            return FirmnessLevel.READY


class MangoLabel(SweetFruitLabel):
    """Label for mango. / 芒果标签。"""
    
    def __init__(self, _init_weight: int, _end_weight: int, _storage_days: int, _firmness: int, _sugar_content: float,
                 _ripeness_state: RipenessState, _comment: str = None):
        super().__init__(_init_weight, _end_weight, _storage_days, _firmness, _sugar_content, _ripeness_state,
                         _comment)

    def get_sugar_level(self) -> SugarLevel:
        """Mango sugar thresholds: <14°Brix (not sweet), 14-16°Brix (ready), >16°Brix (too sweet)"""
        if self.sugar_content < 14:
            return SugarLevel.NOT_SWEET
        elif self.sugar_content > 16:
            return SugarLevel.TOO_SWEET
        else:
            return SugarLevel.READY

    def get_firmness_level(self) -> FirmnessLevel:
        """Mango firmness thresholds: >12500 (hard), 5000-12500 (ready), <5000 (soft)"""
        if self.firmness > 12500:
            return FirmnessLevel.TOO_HARD
        elif self.firmness < 5000:
            return FirmnessLevel.TOO_SOFT
        else:
            return FirmnessLevel.READY


class PapayaLabel(SweetFruitLabel):
    """Label for papaya. / 木瓜标签。"""
    
    def __init__(self, _init_weight: int, _end_weight: int, _storage_days: int, _firmness: int, _sugar_content: float,
                 _ripeness_state: RipenessState, _comment: str = None):
        super().__init__(_init_weight, _end_weight, _storage_days, _firmness, _sugar_content, _ripeness_state,
                         _comment)

    def get_sugar_level(self) -> SugarLevel:
        """Papaya sugar thresholds: <11.5°Brix (not sweet), 11.5-13.5°Brix (ready), >13.5°Brix (too sweet)"""
        if self.sugar_content is None:
            return SugarLevel.UNKNOWN
        elif self.sugar_content < 11.5:
            return SugarLevel.NOT_SWEET
        elif self.sugar_content > 13.5:
            return SugarLevel.TOO_SWEET
        else:
            return SugarLevel.READY

    def get_firmness_level(self) -> FirmnessLevel:
        """Papaya firmness thresholds: >2000 (hard), 1000-2000 (ready), <1000 (soft)"""
        if self.firmness is None:
            return FirmnessLevel.UNKNOWN
        elif self.firmness > 2000:
            return FirmnessLevel.TOO_HARD
        elif self.firmness < 1000:
            return FirmnessLevel.TOO_SOFT
        else:
            return FirmnessLevel.READY


# ==============================================================================
# Fruit Record Class / 水果记录类
# ==============================================================================

class FruitRecord:
    """
    A single fruit measurement record.
    / 单个水果测量记录。
    
    This is the main data structure representing one hyperspectral image
    with all its associated metadata.
    / 这是代表一张高光谱图像及其所有相关元数据的主要数据结构。
    
    Attributes:
        fruit: Fruit type (AVOCADO, KIWI, etc.) / 水果类型
        side: Which side was imaged (FRONT/BACK) / 成像侧面
        day: Measurement day / 测量日期
        id: Unique fruit ID / 唯一水果ID
        camera_type: Which camera was used / 使用的相机
        label: Ground truth label (or None if unlabeled) / 真实标签（如无标签则为None）
    """
    
    def __init__(self, fruit: Fruit, side: Side, day: Day, id: ID, camera_type: CameraType,
                 label: Optional[Union[AvocadoLabel, SweetFruitLabel]] = None):
        self.fruit = fruit
        self.id = id
        self.side = side
        self.day = day
        self.camera_type = camera_type
        self.label = label

    def get_file_path(self):
        """Get the file path for this record. / 获取此记录的文件路径。"""
        return get_file_path(self.fruit, self.side, self.day, self.id, self.camera_type)

    def get_name(self):
        """Get the base name. / 获取基本名称。"""
        return get_name(self.fruit, self.id, self.side, self.day)

    def get_unique_name(self, _postfix: str = None):
        """Get unique name with optional postfix. / 获取带可选后缀的唯一名称。"""
        return get_unique_name(self.fruit, self.id, self.side, self.day, self.camera_type, _postfix)

    def load(self, _origin=None, is_already_referenced=False):
        """
        Load the hyperspectral data for this record.
        / 加载此记录的高光谱数据。
        
        Args:
            _origin: Root data directory / 根数据目录
            is_already_referenced: Whether data is already referenced (white/dark corrected)
                                   数据是否已校正（白/黑校正）
                                   
        Returns:
            (header, data) tuple from spectral_io
        """
        if self.camera_type in (CameraType.VIS, CameraType.NIR,
                                CameraType.VIS_COR):
            if is_already_referenced:
                header, data = spectral_io.load_envi(
                    self.get_file_path(), _origin)
            else:
                header, data = spectral_io.load_referenced_envi(
                    self.get_file_path(), _origin)

            return header, data

    def is_labeled(self):
        """Check if this record has a ground truth label. / 检查此记录是否有真实标签。"""
        return self.label is not None

    def __str__(self):
        if self.is_labeled():
            return f"{self.fruit.value}: {self.id.value}/{self.side.value} \n" \
                f"\ton day: {self.day.value}\n" \
                f"\trecored with: {self.camera_type.value}\n" \
                f"\thas state: {str(self.label)}" \
                f"\tpath: {self.get_file_path()}"
        else:
            return f"{self.fruit.value}: {self.id.value}/{self.side.value} \n" \
                f"\ton day: {self.day.value}\n" \
                f"\trecored with: {self.camera_type.value}\n" \
                f"\tpath: {self.get_file_path()}"

    def __eq__(self, other: Fruit):
        """Equality check. / 相等性检查。"""
        if not isinstance(other, FruitRecord):
            return False

        return (self.fruit == other.fruit) and (self.camera_type == other.camera_type) and (self.side == other.side) and \
               (self.id == other.id) and (self.day == other.day)

    def __hash__(self):
        """Hash for use in sets/dicts. / 用于集合/字典的哈希。"""
        return hash(tuple(sorted(self.__dict__.items())))


# ==============================================================================
# Classification Type / 分类类型
# ==============================================================================

class ClassificationType(enum.Enum):
    """
    Type of classification task.
    / 分类任务类型。
    
    RIPENESS: 3-class ripeness classification (unripe/perfect/overripe)
              3类成熟度分类
    FIRMNESS: 3-class firmness classification (too_hard/ready/too_soft)
              3类硬度分类
    SUGAR: 3-class sugar content classification (not_sweet/ready/too_sweet)
           3类糖度分类
    """
    RIPENESS = 'ripeness'
    FIRMNESS = 'firmness'
    SUGAR = 'sugar'


def label2text(l, _english=False):
    """
    Convert label to display text.
    / 将标签转换为显示文本。
    
    Args:
        l: Label enum value / 标签枚举值
        _english: If True, return English; otherwise return German
                  如果为True返回英文，否则返回德语
    """
    if not _english:
        # German labels / 德语标签
        if l == RipenessState.UNRIPE:
            return 'Unreif'
        if l == RipenessState.RIPE:
            return 'Reif'
        if l == RipenessState.PERFECT:
            return 'Perfekt'
        if l == RipenessState.OVERRIPE:
            return 'Überreif'
        if l == RipenessState.NEAR_OVERRIPE:
            return 'Nahe überreif'
        if l == FirmnessLevel.TOO_HARD:
            return 'Zu hart'
        if l == FirmnessLevel.READY:
            return 'Gut'
        if l == FirmnessLevel.TOO_SOFT:
            return 'Zu weich'
        if l == SugarLevel.NOT_SWEET:
            return 'Nicht süß'
        if l == SugarLevel.READY:
            return 'Gut'
        if l == SugarLevel.TOO_SWEET:
            return 'Zu süß'
    else:
        # English labels / 英语标签
        if l == RipenessState.UNRIPE:
            return 'Unripe'
        if l == RipenessState.RIPE:
            return 'Ripe'
        if l == RipenessState.PERFECT:
            return 'Ready'
        if l == RipenessState.OVERRIPE:
            return 'Overripe'
        if l == RipenessState.NEAR_OVERRIPE:
            return 'Near overripe'
        if l == FirmnessLevel.TOO_HARD:
            return 'Too hard'
        if l == FirmnessLevel.READY:
            return 'Ready'
        if l == FirmnessLevel.TOO_SOFT:
            return 'Too soft'
        if l == SugarLevel.NOT_SWEET:
            return 'Not sweet'
        if l == SugarLevel.READY:
            return 'Ready'
        if l == SugarLevel.TOO_SWEET:
            return 'Too sweet'
