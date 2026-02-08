import torch
import torch.nn as nn
from torchvision.models import resnet18

class ResNet18Cifar(nn.Module):
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.backbone = resnet18(weights=None)
        # adapt first conv for CIFAR (3x32x32)
        self.backbone.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.backbone.maxpool = nn.Identity()
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)
