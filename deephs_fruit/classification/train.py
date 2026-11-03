"""
================================================================================
DeepHS-Fruit Training Script / 高光谱水果成熟度训练脚本
================================================================================
This is the MAIN training script for the paper:
"Measuring the Ripeness of Fruit with Hyperspectral Imaging and Deep Learning"
(IJCNN 2021, Varga et al.)

这是论文的主训练脚本：
"基于高光谱成像与深度学习的水果成熟度测量" (IJCNN 2021)

PURPOSE / 用途:
    Train a deep neural network to classify fruit ripeness/firmness/sugar
    from hyperspectral images (3D data: height x width x spectral bands).
    
    训练深度神经网络，从高光谱图像（3D数据：高×宽×光谱波段）
    分类水果的成熟度/硬度/糖度。

INPUT / 输入:
    - Hyperspectral cubes (.hdr + .bin files) / 高光谱立方体数据
    - Labels from destructive measurements / 破坏性测量得到的标签
    
OUTPUT / 输出:
    - Trained model checkpoint (.ckpt) / 训练好的模型检查点
    - Test accuracy, confusion matrix / 测试准确率、混淆矩阵

KEY CONCEPTS / 关键概念:
    - Hyperspectral Image: Like a normal photo but with 224+ color channels
      (not just RGB). Each pixel has a full spectrum.
      高光谱图像：像普通照片，但有224+个颜色通道（不只是RGB）。
      每个像素都有一条完整的光谱曲线。
    
    - HS-CNN (Hyperspectral CNN): 3D convolutional network that processes
      both spatial (image) and spectral (wavelength) information together.
      高光谱CNN：3D卷积网络，同时处理空间（图像）和光谱（波长）信息。
================================================================================
"""

import argparse
import sys

import pytorch_lightning as lightning
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers.wandb import WandbLogger
import torch
from torch.utils.data import DataLoader
from torchvision import transforms 
import wandb

from classification.model_factory import get_model, VALID_MODELS, CAMERA_AGNOSTIC_MODELS
from classification.models.hyve.hyve_convolution import HyVEConv
from classification.transformers.data_augmentation import Augmenter
from classification.transformers.normalize import Normalize
from classification.utils.confusion_matrix import log_confusion_matrix
import core.argparser_utils as argparser_utils
from core.datasets.hyperspectral_dataset import HyperspectralDataset, get_records
from core.lightning_callbacks.lr_logger import LRLoggingCallback
from core.name_convention import CameraType, ClassificationType, Fruit
from core.run_utils import (
    get_current_git_hash,
    get_slurm_job_id,
    get_slurm_job_path,
    get_wandb_log_dir,
)
from core.loss.focalloss import FocalLoss
import core.util as util

# 尝试导入AdaBound优化器（论文推荐使用）
# Try to import AdaBound optimizer (recommended by paper)
try:
    import adabound
    ADABOUND_AVAILABLE = True
except ImportError:
    ADABOUND_AVAILABLE = False
    print("Warning: AdaBound not available, falling back to SGD / 警告：AdaBound不可用，回退到SGD")

# ==============================================================================
# AUGMENTATION CONFIGURATIONS / 数据增强配置
# ==============================================================================
# Data augmentation: Artificially create more training data by transforming
# existing images (flip, rotate, crop, etc.). Helps prevent overfitting.
# 数据增强：通过对现有图像进行变换（翻转、旋转、裁剪等）来人工创建更多训练数据。
# 有助于防止过拟合。

AUGMENTATION_CONFIG_TRAIN = {
    'random_flip': True,      # Randomly flip image horizontally / 随机水平翻转
    'random_rotate': True,    # Randomly rotate image / 随机旋转
    'random_noise': False,    # Don't add noise (can destabilize) / 不加噪声（可能不稳定）
    'random_cut': True,       # Random cutout (hide parts of image) / 随机遮挡
    'random_crop': True,      # Random cropping / 随机裁剪
    'random_intensity_scale': False  # Don't scale intensity / 不缩放强度
}

# TTA = Test Time Augmentation / 测试时增强
# During testing, we make multiple predictions with different augmentations
# and average them. This improves accuracy but takes longer.
# 测试时，我们用不同的增强做多次预测然后取平均。提高准确率但更耗时。
AUGMENTATION_CONFIG_TTA = {
    'random_flip': True,
    'random_rotate': True,
    'random_noise': False,
    'random_cut': False,      # No cutout during test / 测试时不遮挡
    'random_crop': False,     # No crop during test / 测试时不裁剪
    'random_intensity_scale': False
}


# ==============================================================================
# DeepHsModule: Main Training Module / 主训练模块
# ==============================================================================
# This class defines the entire training process using PyTorch Lightning.
# It inherits from LightningModule which handles the training loop automatically.
# 这个类使用 PyTorch Lightning 定义整个训练过程。
# 它继承自 LightningModule，自动处理训练循环。

class DeepHsModule(lightning.LightningModule):
    """
    Main training module for DeepHS-Fruit.
    / DeepHS-Fruit 的主训练模块。
    
    This class manages:
    / 这个类管理：
    - Model architecture (HS-CNN) / 模型架构
    - Data loading (train/val/test splits) / 数据加载
    - Training step (forward pass, loss calculation) / 训练步骤
    - Validation step / 验证步骤
    - Test step (with confusion matrix) / 测试步骤
    - Optimization (SGD with learning rate scheduling) / 优化
    """
    
    def __init__(self, hparams):
        """
        Initialize the module with hyperparameters.
        / 用超参数初始化模块。
        
        Args:
            hparams: Dictionary of hyperparameters (lr, batch_size, etc.)
                     超参数字典（学习率、批次大小等）
        """
        super(DeepHsModule, self).__init__()
        self.save_hyperparameters(hparams)  # Save all hparams for logging / 保存所有超参数用于日志
        self.model = get_model(self.hparams)  # Get the neural network model / 获取神经网络模型
        
        # 根据配置选择损失函数（论文使用Focal Loss处理类别不平衡）
        # Select loss function based on config (paper uses Focal Loss for class imbalance)
        if self.hparams.get('use_focal_loss', True):
            # Focal Loss: 论文消融实验显示CE只有80%，Focal Loss到93.3%
            # gamma=2.0是论文常用值，解决unripe样本过多的问题
            self.critertion = FocalLoss(gamma=2.0, size_average=True)
            print("Using Focal Loss (gamma=2.0) / 使用Focal Loss (gamma=2.0)")
        else:
            self.critertion = torch.nn.CrossEntropyLoss()
            print("Using CrossEntropyLoss / 使用CrossEntropyLoss")
        
        self.is_camera_agnostic = self.hparams['model'] in CAMERA_AGNOSTIC_MODELS
        self.train_records, self.val_records, self.test_records = None, None, None
        self.train_dataset, self.val_dataset, self.test_dataset = None, None, None
        self.test_step_outputs = []  # Store test outputs for aggregation / 存储测试输出用于汇总

    def setup(self, stage: str):
        """
        Prepare datasets for training, validation, or testing.
        / 为训练、验证或测试准备数据集。
        
        Args:
            stage: 'fit' (train+val) or 'test' / 'fit'（训练+验证）或 'test'
        """
        # Get train/val/test record splits / 获取训练/验证/测试记录划分
        self.train_records, self.val_records, self.test_records = \
            get_records(self.hparams['fruit'],
                        self.hparams['camera_type'],
                        self.hparams['classification_type'],
                        use_inter_ripeness_levels=True,
                        extend_by_time_assumption=True,
                        allow_all_fruit_types=True,
                        random_split=self.hparams.get('random_split', False),
                        seed=self.hparams.get('seed', 42)
                        )

        # Normalization: Scale data to have mean=0, std=1
        # 归一化：将数据缩放到均值为0，标准差为1
        common_preprocessing = [Normalize(self.hparams['camera_type'])]

        if stage == 'fit':
            # Training data: apply augmentation + normalization
            # 训练数据：应用数据增强 + 归一化
            train_preprocessing = []
            train_preprocessing += [Augmenter(augmentation_config=self.hparams['augmentation_config'],
                                              input_size=self.hparams['input_size'])]
            self.train_dataset = HyperspectralDataset(self.hparams['classification_type'], self.train_records,
                                                      data_path=self.hparams['data_path'],
                                                      balance=True,  # Balance class sizes / 平衡类别数量
                                                      transform=transforms.Compose(
                                                          train_preprocessing + common_preprocessing),
                                                      input_size=self.hparams['input_size'])
            # Validation data: only normalization (no augmentation)
            # 验证数据：只归一化（不增强）
            self.val_dataset = HyperspectralDataset(self.hparams['classification_type'],
                                                    self.val_records,
                                                    data_path=self.hparams['data_path'],
                                                    transform=transforms.Compose(
                                                        common_preprocessing),
                                                    input_size=self.hparams['input_size'])
        elif stage == 'test':
            # Test data: only normalization
            # 测试数据：只归一化
            self.test_dataset = HyperspectralDataset(self.hparams['classification_type'], self.test_records,
                                                     data_path=self.hparams['data_path'],
                                                     transform=transforms.Compose(
                                                         common_preprocessing),
                                                     input_size=self.hparams['input_size'])

    def train_dataloader(self) -> DataLoader:
        """Create training data loader / 创建训练数据加载器"""
        return DataLoader(self.train_dataset, self.hparams['batch_size'], num_workers=self.hparams['num_workers'],
                          shuffle=True, drop_last=True, collate_fn=None)

    def val_dataloader(self) -> DataLoader:
        """Create validation data loader / 创建验证数据加载器"""
        return DataLoader(self.val_dataset, 1, num_workers=self.hparams['num_workers'],
                          shuffle=False)

    def test_dataloader(self):
        """Create test data loader / 创建测试数据加载器"""
        return DataLoader(self.test_dataset, 1, num_workers=self.hparams['num_workers'])

    def forward(self, x, channel_wavelengths, test_augmentation=True):
        """
        Forward pass: compute predictions from input data.
        / 前向传播：从输入数据计算预测结果。
        
        Args:
            x: Input hyperspectral image (batch, bands, height, width)
               输入高光谱图像 (批次, 波段, 高, 宽)
            channel_wavelengths: Wavelength values for each band / 每个波段的波长值
            test_augmentation: Whether to use TTA / 是否使用测试时增强
        """
        if test_augmentation:
            # TTA: Make multiple predictions and average / 测试时增强：多次预测取平均
            augmentation_config = self.hparams['tta_augmentation_config']
            augmenter = Augmenter(augmentation_config=augmentation_config,
                                  input_size=self.hparams['input_size'])

            iterations = self.hparams['tta_augmentation_iterations']

            preds = torch.zeros(
                (iterations, len(x), self.hparams['num_classes']))

            for i in range(iterations):
                # the augmentation should not change the label, so we can ignore it here.
                # 增强不应改变标签，所以这里可以忽略标签。
                _batch = [augmenter((_x, _y, _channel_wavelengths)) for _x, _y, _channel_wavelengths in zip(
                    x.clone(), torch.zeros((x.shape[0], 1)), channel_wavelengths)]
                _x = torch.stack(list(zip(*_batch))[0])
                _channel_wavelengths = list(zip(*_batch))[2]
                if isinstance(_channel_wavelengths, torch.Tensor):
                    _channel_wavelengths = torch.stack(_channel_wavelengths)

                preds[i] = self.model(
                    _x, channel_wavelengths=list(_channel_wavelengths))

            pred = preds.mean(0)  # Average predictions / 平均预测结果
        else:
            pred = self.model(x,
                              channel_wavelengths=channel_wavelengths)

        return pred

    @torch.no_grad()
    def predict(self, x, test_augmentation=True):
        """
        Predict class labels for input images.
        / 预测输入图像的类别标签。
        """
        return self.forward(x.cuda(), test_augmentation=test_augmentation).argmax(-1).detach().cpu()

    def training_step(self, batch, batch_id):
        """
        Single training step: forward + loss + log.
        / 单个训练步骤：前向传播 + 计算损失 + 记录。
        """
        x, y, channel_wavelengths = batch

        pred = self.model(x, channel_wavelengths=channel_wavelengths)
        loss = self.critertion(pred, y)  # Cross-entropy loss / 交叉熵损失
        assert torch.isnan(loss) == False
        accuracy = ((pred.argmax(1) == y).sum().float() / x.shape[0]).detach()
        self.log("train/loss", loss, on_step=True)
        self.log("train/accuracy", accuracy, on_step=True)

        if self.is_camera_agnostic:
            conv = self._get_hyve_conv()
            if conv is not None:
                for cw in channel_wavelengths:
                    _, activations = conv._get_gauss_features_and_activation(cw)
                for i, m in enumerate(activations):
                    self.log(
                        f"hyve_conv/gauss_activations_{i}", m, rank_zero_only=True)

        return loss

    def validation_step(self, batch, batch_id):
        """
        Single validation step.
        / 单个验证步骤。
        """
        x, y, channel_wavelengths = batch

        pred = self.model(x, channel_wavelengths=channel_wavelengths)
        loss = self.critertion(pred, y)

        accuracy = ((pred.argmax(1) == y).sum().float() / x.shape[0]).detach()

        self.log("val/loss", loss,
                 on_epoch=True, sync_dist=True)
        self.log("val/accuracy",
                 accuracy, on_epoch=True, sync_dist=True)

    def _get_hyve_conv(self) -> HyVEConv:
        """Get HyVE convolution layer (for camera-agnostic models) / 获取HyVE卷积层"""
        conv = None
        if self.hparams['model'] in ('hyve'):
            conv = self.model.get_hyve_conv()

        return conv

    def on_validation_epoch_end(self):
        """
        Called at end of each validation epoch.
        / 每个验证轮次结束时调用。
        """
        if self.is_camera_agnostic and self.trainer.is_global_zero:
            conv = self._get_hyve_conv()

            if conv is not None:
                gauss_mean, gauss_variance = conv.get_gauss().scaled_params()
                for i, m in enumerate(gauss_mean):
                    self.log(
                        f"hyve_conv/gauss_mean_{i}", m, rank_zero_only=True)
                for i, v in enumerate(gauss_variance):
                    self.log(
                        f"hyve_conv/gauss_variance_{i}", v, rank_zero_only=True)

                # log training progress of the hyve convolutions
                if self.current_epoch % 20 == 0:
                    if conv.share_features:
                        kernels_channelwise, kernels_gaussianwise, kernels_convwise = None, None, None
                        kernel_prototypes = conv._get_unweighted_kernels_shared()
                        if len(kernel_prototypes) == 3:
                            kernels_channelwise, kernels_gaussianwise, kernels_convwise = kernel_prototypes
                        else:
                            kernels_channelwise, kernels_convwise = kernel_prototypes

                        if kernels_channelwise is not None:
                            kernels_channelwise = kernels_channelwise.reshape(-1,
                                                                              kernels_channelwise.shape[-2],
                                                                              kernels_channelwise.shape[-1])
                            self.logger.experiment.log({f'hyve_conv/unweighted_kernels_channelwise':
                                                            [wandb.Image(kernel) for kernel in kernels_channelwise]})
                        if kernels_gaussianwise is not None:
                            kernels_gaussianwise = kernels_gaussianwise.reshape(-1,
                                                                                kernels_gaussianwise.shape[-2],
                                                                                kernels_gaussianwise.shape[-1])
                            self.logger.experiment.log({f'hyve_conv/unweighted_kernels_gaussianwise':
                                                            [wandb.Image(kernel) for kernel in kernels_gaussianwise]})
                        if kernels_convwise is not None:
                            kernels_convwise = kernels_convwise.reshape(-1,
                                                                        kernels_convwise.shape[-2],
                                                                        kernels_convwise.shape[-1])
                            self.logger.experiment.log({f'hyve_conv/unweighted_kernels_convwise':
                                                            [wandb.Image(kernel) for kernel in kernels_convwise]})

                        kernels = conv._get_unweighted_kernels_unmerged()
                        kernels = kernels.reshape(-1,
                                                  kernels.shape[-2], kernels.shape[-1])
                        self.logger.experiment.log({f'hyve_conv/unweighted_kernels_individual':
                                                        [wandb.Image(kernel) for kernel in kernels]})
                        for i, v in enumerate(conv.get_kernel_prototype_share_factors()):
                            if v is not None:
                                self.log(
                                    f"hyve_conv/kernel_share_impact_{i}", v, rank_zero_only=True)

                    kernels = conv.get_unweighted_kernels()
                    kernels = conv.get_unweighted_kernels()
                    kernels = kernels.reshape(-1,
                                              kernels.shape[-2], kernels.shape[-1])
                    self.logger.experiment.log({f'hyve_conv/unweighted_kernels':
                                                    [wandb.Image(kernel) for kernel in kernels]})

    def test_step(self, batch, batch_id):
        """
        Single test step.
        / 单个测试步骤。
        """
        x, y, channel_wavelengths = batch

        pred = self.forward(x,
                            channel_wavelengths=channel_wavelengths,
                            test_augmentation=self.hparams['test_augmentation'])
        pred = pred.to(y.device)
        loss = self.critertion(pred, y)
        accuracy = ((pred.argmax(1) == y).sum().float() / x.shape[0]).detach()

        output = {
            'test_loss': loss,
            'test_accuracy': accuracy,
            'test_ys': torch.stack(
                [torch.tensor([y_pred, y_true]) for y_pred, y_true in zip(pred.argmax(1), y)])
        }
        self.test_step_outputs.append(output)
        return output

    def on_test_epoch_end(self):
        """
        Called at end of test epoch: aggregate results and log.
        / 测试轮次结束时调用：汇总结果并记录。
        """
        if hasattr(self, 'test_step_outputs'):
            outputs = self.all_gather(self.test_step_outputs)
            test_accuracy = torch.stack([o['test_accuracy']
                                         for o in outputs]).mean()
            test_loss = torch.stack([o['test_loss']
                                     for o in outputs]).mean()
            test_ys = torch.cat([o['test_ys']
                                 for o in outputs]).reshape(-1, 2).cpu().numpy()

            # log only in the first worker / 只在第一个工作进程记录
            if self.trainer.is_global_zero:
                self.log('test/accuracy',
                         test_accuracy, rank_zero_only=True)
                self.log('test/loss',
                         test_loss, rank_zero_only=True)

                # Set labels based on classification type / 根据分类类型设置标签
                labels = []
                if self.hparams['classification_type'] == ClassificationType.RIPENESS:
                    labels = ["Unripe", "Ripe", "Overripe"]
                elif self.hparams['classification_type'] == ClassificationType.FIRMNESS:
                    labels = ["Too hard", "Perfect", "Too soft"]
                elif self.hparams['classification_type'] == ClassificationType.SUGAR:
                    labels = ["Not sweet", "Perfect", "Too sweet"]

                log_confusion_matrix(
                    self.logger, test_ys[:, 0], test_ys[:, 1], labels=labels)
            
            # Clear collected outputs / 清空收集的输出
            self.test_step_outputs.clear()

    def configure_optimizers(self):
        """
        Configure optimizer and learning rate scheduler.
        / 配置优化器和学习率调度器。
        
        Paper uses AdaBound with lr=0.01 (key for 93.3% accuracy).
        Ablation study shows: SGD/Adam only reach ~80%, AdaBound reaches 93.3%.
        论文使用AdaBound+lr=0.01（达到93.3%准确率的关键）。
        消融实验显示：SGD/Adam只能到~80%，AdaBound能到93.3%。
        """
        # 使用AdaBound优化器（如果可用），否则回退到SGD
        # Use AdaBound optimizer if available, otherwise fallback to SGD
        if ADABOUND_AVAILABLE and self.hparams.get('use_adabound', True):
            self.optimizer = adabound.AdaBound(
                self.model.parameters(),
                lr=self.hparams['lr'],  # 论文推荐lr=0.01 / paper recommends lr=0.01
                final_lr=0.1,  # AdaBound特有参数 / AdaBound specific parameter
                weight_decay=0.0001
            )
            print(f"Using AdaBound optimizer (lr={self.hparams['lr']}) / 使用AdaBound优化器")
        else:
            self.optimizer = torch.optim.SGD(
                self.model.parameters(), 
                lr=self.hparams['lr'],
                momentum=0.9,
                weight_decay=0.0001)
            print(f"Using SGD optimizer (lr={self.hparams['lr']}) / 使用SGD优化器")
        
        # 学习率调度：每10轮下降50%
        # Learning rate schedule: decay by 50% every 10 epochs
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer, 
            step_size=self.hparams.get('lr_step_size', 10), 
            gamma=self.hparams.get('lr_gamma', 0.5),
            verbose=True)

        return {
            'optimizer': self.optimizer,
            'lr_scheduler': self.scheduler,
            'monitor': 'val_loss'
        }


# ==============================================================================
# Command Line Argument Parser / 命令行参数解析器
# ==============================================================================

def get_parser():
    """
    Define command-line arguments for training.
    / 定义训练的命令行参数。
    """
    def str2bool(v):
        """Convert string to boolean / 将字符串转换为布尔值"""
        if isinstance(v, bool):
            return v
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected.')

    parser = argparse.ArgumentParser(
        "DeepHS training:")
    parser.add_argument("--model", default="deephs_net", type=str,
                        choices=VALID_MODELS.keys())
    parser.add_argument("--batch_size", type=int, default=64,
                        help="The number of images per batch / 每批次的图像数量")
    parser.add_argument("--lr", type=float, default=1e-2,
                        help="Learning rate / 学习率")
    parser.add_argument(
        '--fruit', type=argparser_utils.str2fruit, default=Fruit.AVOCADO,
        choices=Fruit,
        help="Fruit type / 水果类型")
    parser.add_argument(
        '--camera_type', type=argparser_utils.str2cameratype, default=CameraType.VIS,
        help="Camera type (vis/nir/vis_cor) / 相机类型")
    parser.add_argument('--classification_type', type=argparser_utils.str2classification_type,
                        default=ClassificationType.FIRMNESS,
                        help="Classification task (ripeness/firmness/sugar) / 分类任务")
    parser.add_argument("--num_epochs", type=int, default=5000,
                        help="Maximum training epochs / 最大训练轮数")
    parser.add_argument("--data_path", type=str, required=True,
                        help="the root folder of dataset / 数据集根目录")
    parser.add_argument("--seed", type=int, default=0,
                        help="Random seed for reproducibility / 随机种子")
    parser.add_argument("--log_path", type=str, default=None)
    parser.add_argument("--online_logging", default=False, action='store_true')
    parser.add_argument("--debug", default=False, action='store_true')
    parser.add_argument("--model_checkpoint", default=None, type=str)

    parser.add_argument("--comment", default=None, type=str)
    parser.add_argument("--camera_agnostic_num_gauss", default=5, type=int,
                        help='The number of gaussian distributions used for the camera agnostic convolution. Parameter G')
    parser.add_argument("--random_split", default=False, action='store_true',
                        help="Use random split by fruit ID instead of hardcoded split / 按水果ID随机划分代替硬编码划分")
    
    # 论文关键配置参数 / Paper key configuration parameters
    parser.add_argument("--use_focal_loss", default=True, type=str2bool,
                        help="Use Focal Loss (paper: 93.3% vs CE: 80%) / 使用Focal Loss")
    parser.add_argument("--use_adabound", default=True, type=str2bool,
                        help="Use AdaBound optimizer (paper: 93.3% vs SGD: 80%) / 使用AdaBound优化器")
    parser.add_argument("--lr_step_size", default=10, type=int,
                        help="LR scheduler step size (epochs) / 学习率调度步长")
    parser.add_argument("--lr_gamma", default=0.5, type=float,
                        help="LR decay factor / 学习率衰减系数")
    
    # 模型容量配置（解决欠拟合）/ Model capacity config (fix underfitting)
    parser.add_argument("--hidden_layers", default="64,128,256", type=str,
                        help="Hidden layer sizes (comma-separated) / 隐藏层大小（逗号分隔）")

    return parser


def get_args(parser_generator=get_parser):
    """Parse command-line arguments / 解析命令行参数"""
    args = parser_generator().parse_args()

    if args.fruit == Fruit.AVOCADO and args.classification_type == ClassificationType.SUGAR:
        print("! Avocados have no sugar label! / 牛油果没有糖度标签！")
        exit(0)

    # 解析hidden_layers字符串为列表 / Parse hidden_layers string to list
    if isinstance(args.hidden_layers, str):
        args.hidden_layers = [int(x.strip()) for x in args.hidden_layers.split(',')]
    
    print(f"Model hidden layers: {args.hidden_layers}")

    return args


# ==============================================================================
# Main Training Function / 主训练函数
# ==============================================================================

def main(hparams):
    """
    Main training function.
    / 主训练函数。
    
    This function:
    / 这个函数：
    1. Sets up hyperparameters / 设置超参数
    2. Creates the model / 创建模型
    3. Sets up training callbacks (early stopping, checkpointing) / 设置训练回调
    4. Trains the model / 训练模型
    5. Tests the best model / 测试最佳模型
    """
    hparams['type'] = 'single_camera'
    hparams['git_id'] = get_current_git_hash()
    hparams['slurm_job_id'] = get_slurm_job_id()
    hparams['slurm_job_path'] = get_slurm_job_path()
    # Windows 上使用多进程会导致内存问题，设为 0
    import platform
    if platform.system() == 'Windows':
        hparams['num_workers'] = 0
    else:
        hparams['num_workers'] = 4 if not hparams['debug'] else 0

    hparams['bands'] = len(util.get_wavelengths_for(hparams['camera_type']))
    hparams['wavelengths'] = util.get_wavelengths_for(hparams['camera_type'])
    hparams['augmentation_config'] = AUGMENTATION_CONFIG_TRAIN
    hparams['test_augmentation'] = True
    hparams['tta_augmentation_config'] = AUGMENTATION_CONFIG_TTA
    hparams['tta_augmentation_iterations'] = 5
    hparams['num_classes'] = 3
    hparams['input_size'] = (64, 64)
    
    # 论文关键配置 / Paper key configurations
    # 默认使用Focal Loss和AdaBound（论文达到93.3%的关键）
    hparams['use_focal_loss'] = hparams.get('use_focal_loss', True)
    hparams['use_adabound'] = hparams.get('use_adabound', True)
    hparams['lr_step_size'] = hparams.get('lr_step_size', 10)
    hparams['lr_gamma'] = hparams.get('lr_gamma', 0.5)
    
    # 解析hidden_layers参数 / Parse hidden_layers parameter
    if isinstance(hparams.get('hidden_layers'), str):
        hparams['hidden_layers'] = [int(x.strip()) for x in hparams['hidden_layers'].split(',')]
    print(f"Model hidden layers: {hparams['hidden_layers']}")

    print("Hparams: %s" % hparams)

    model = DeepHsModule(hparams)
    logger = WandbLogger(offline=not hparams['online_logging'], save_dir=hparams['log_path'],
                         project='deephs') if 'logger' not in hparams.keys() else hparams['logger']

    # Early stopping: Stop if validation loss doesn't improve for 50 epochs
    # 早停：如果验证损失50轮不改善就停止（给模型更多收敛时间）
    early_stop_callback = EarlyStopping(
        monitor='val/loss',
        min_delta=0.001,  # 需要至少0.001的改善才算有效
        verbose=True,
        mode='min',
        patience=50
    )

    # Checkpoint: Save the best model based on validation loss
    # 检查点：基于验证损失保存最佳模型
    checkpoint_callback = ModelCheckpoint(
        filename='best',
        save_top_k=1,
        verbose=True,
        monitor='val/loss',
        mode='min'
    )

    # Trainer: Manages the entire training process
    # 训练器：管理整个训练过程
    trainer = lightning.Trainer(max_epochs=opt.num_epochs,
                                accelerator='gpu',
                                devices=1,
                                logger=logger,
                                strategy='auto',
                                min_epochs=50,
                                callbacks=[LRLoggingCallback(),
                                           early_stop_callback,
                                           checkpoint_callback
                                           ],
                                log_every_n_steps=1,
                                num_sanity_val_steps=0,
                                gradient_clip_val=1.0)  # Prevent gradient explosion / 防止梯度爆炸

    trainer.fit(model)
    best_model = DeepHsModule.load_from_checkpoint(
        checkpoint_callback.best_model_path)
    best_model.eval()

    print(f"Best model [{checkpoint_callback.best_model_path}]..")
    result = trainer.test(best_model)

    return result[0]


if __name__ == "__main__":
    # Windows encoding fix / Windows编码修复
    if sys.platform == "win32":
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    opt = get_args()
    num_gpus = torch.cuda.device_count()

    # fix the seed for reproducibility / 固定随机种子以便复现
    seed = opt.seed
    lightning.seed_everything(seed)

    if get_wandb_log_dir() is not None:
        opt.log_path = get_wandb_log_dir()

    hparams = vars(opt)

    main(hparams)
