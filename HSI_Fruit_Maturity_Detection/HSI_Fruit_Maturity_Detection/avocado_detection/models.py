import torch
import torch.nn as nn


class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        mid = max(channels // reduction, 4)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc  = nn.Sequential(
            nn.Linear(channels, mid), nn.ReLU(True),
            nn.Linear(mid, channels), nn.Sigmoid()
        )

    def forward(self, x):
        w = self.fc(self.gap(x).flatten(1)).unsqueeze(-1).unsqueeze(-1)
        return x * w


class ClassifierNetwork(nn.Module):
    def __init__(self, bands, num_classes=3, hidden_layers=[25, 30, 50]):
        super().__init__()
        k = 3
        self.block1 = nn.Sequential(
            nn.Conv2d(bands, bands * k, 7, padding=3, groups=bands),
            nn.Conv2d(bands * k, hidden_layers[0], 1),
            nn.ReLU(True), nn.AvgPool2d(2),
            nn.BatchNorm2d(hidden_layers[0]),
            SEBlock(hidden_layers[0]),
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(hidden_layers[0], hidden_layers[0]*k, 3, padding=1, groups=hidden_layers[0]),
            nn.Conv2d(hidden_layers[0]*k, hidden_layers[1], 1),
            nn.ReLU(True), nn.AvgPool2d(2),
            nn.BatchNorm2d(hidden_layers[1]),
            SEBlock(hidden_layers[1]),
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(hidden_layers[1], hidden_layers[1]*k, 3, padding=1, groups=hidden_layers[1]),
            nn.Conv2d(hidden_layers[1]*k, hidden_layers[2], 1),
            nn.ReLU(True), nn.BatchNorm2d(hidden_layers[2]),
        )
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc  = nn.Sequential(
            nn.Linear(hidden_layers[2], hidden_layers[2]),
            nn.BatchNorm1d(hidden_layers[2]),
            nn.ReLU(True), nn.Dropout(0.3),
            nn.Linear(hidden_layers[2], num_classes)
        )
        self._init_weights()
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None: nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                nn.init.ones_(m.weight); nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None: nn.init.zeros_(m.bias)
    def forward(self, x):
        return self.fc(self.gap(self.block3(self.block2(self.block1(x)))).flatten(1))
