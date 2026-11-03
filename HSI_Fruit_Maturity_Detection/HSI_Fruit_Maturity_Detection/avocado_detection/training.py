import copy
import numpy as np
import torch


class EMA:
    def __init__(self, model, init_decay=0.9):
        self.decay  = init_decay
        self.shadow = copy.deepcopy(model)
        self.shadow.eval()
        for p in self.shadow.parameters():
            p.requires_grad_(False)
    @torch.no_grad()
    def update(self, model):
        for s_p, m_p in zip(self.shadow.parameters(), model.parameters()):
            s_p.data.mul_(self.decay).add_(m_p.data, alpha=1.0 - self.decay)
        for s_b, m_b in zip(self.shadow.buffers(), model.buffers()):
            s_b.copy_(m_b)
    def set_decay(self, decay):
        self.decay = decay
    def eval_model(self):
        return self.shadow


class WarmupCosineScheduler:
    def __init__(self, optimizer, warmup_epochs, total_epochs, base_lr, min_lr=1e-6):
        self.optimizer     = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs  = total_epochs
        self.base_lr       = base_lr
        self.min_lr        = min_lr
        self.current_epoch = 0
    def step(self):
        e = self.current_epoch
        if e < self.warmup_epochs:
            lr = self.base_lr * (e + 1) / self.warmup_epochs
        else:
            progress = (e - self.warmup_epochs) / max(
                self.total_epochs - self.warmup_epochs, 1)
            lr = self.min_lr + 0.5 * (self.base_lr - self.min_lr) * (
                1 + np.cos(np.pi * progress))
        for pg in self.optimizer.param_groups:
            pg['lr'] = lr
        self.current_epoch += 1
        return lr


def augment_batch(x):
    out = x.clone()
    for i in range(len(out)):
        if torch.rand(1) > 0.5: out[i] = torch.flip(out[i], [1])
        if torch.rand(1) > 0.5: out[i] = torch.flip(out[i], [2])
        if torch.rand(1) > 0.5: out[i] = torch.rot90(out[i], 1, [1, 2])
        if torch.rand(1) > 0.5: out[i] += torch.randn_like(out[i]) * 0.02
    return out
