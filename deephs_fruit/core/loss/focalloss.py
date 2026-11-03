"""
Focal Loss for DeepHS-Fruit
Reference: "Focal Loss for Dense Object Detection" (Lin et al., 2017)

Focal Loss addresses class imbalance by down-weighting easy examples 
and focusing on hard negatives. Critical for the highly imbalanced 
DeepHS-Fruit dataset (unripe samples dominate).

Focal Loss公式：
FL(p_t) = -α_t * (1 - p_t)^γ * log(p_t)

其中：
- p_t: 模型对正确类别的预测概率
- α_t: 类别权重（解决类别不平衡）
- γ (gamma): 聚焦参数，γ>0时降低易分类样本的权重
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Focal Loss implementation for multi-class classification.
    / 多分类Focal Loss实现。
    
    Args:
        alpha: 类别权重，可以是float或tensor
               Class weight, can be float or tensor
        gamma: 聚焦参数，默认2.0（论文推荐值）
               Focusing parameter, default 2.0 (paper recommendation)
        size_average: 是否对batch取平均
                      Whether to average over batch
        reduce: 是否降维
                Whether to reduce dimensions
    """
    
    def __init__(self, alpha=None, gamma=2.0, size_average=True, reduce=True):
        super(FocalLoss, self).__init__()
        self.alpha = alpha  # 类别权重 / class weights
        self.gamma = gamma  # 聚焦参数 / focusing parameter
        self.size_average = size_average
        self.reduce = reduce
        
    def forward(self, inputs, targets):
        """
        Compute focal loss.
        / 计算focal loss。
        
        Args:
            inputs: 模型输出logits [batch_size, num_classes]
            targets: 真实标签 [batch_size]
        
        Returns:
            Focal loss scalar
        """
        # Convert inputs to probabilities using softmax
        # 使用softmax将logits转为概率
        probs = F.softmax(inputs, dim=1)
        
        # Get probability of the true class for each sample
        # 获取每个样本对应真实类别的概率
        batch_size = inputs.size(0)
        p_t = probs.gather(1, targets.view(-1, 1)).view(-1)
        
        # Compute focal weight: (1 - p_t)^gamma
        # 计算focal权重
        focal_weight = torch.pow(1.0 - p_t, self.gamma)
        
        # Compute cross-entropy loss (without reduction)
        # 计算交叉熵损失（不降维）
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        
        # Apply focal weight
        # 应用focal权重
        focal_loss = focal_weight * ce_loss
        
        # Apply alpha (class weighting) if provided
        # 如果提供了alpha，应用类别权重
        if self.alpha is not None:
            if isinstance(self.alpha, (list, tuple)):
                # Convert to tensor / 转为tensor
                alpha_t = torch.tensor(self.alpha, device=inputs.device)[targets]
            else:
                alpha_t = self.alpha
            focal_loss = alpha_t * focal_loss
        
        # Reduce loss
        # 降维损失
        if self.reduce:
            if self.size_average:
                return focal_loss.mean()
            else:
                return focal_loss.sum()
        else:
            return focal_loss
