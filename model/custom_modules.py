"""
自定义模块：用于 YOLO11 轻量化改进
包含 ECA (Efficient Channel Attention) 和 Slim-Neck 相关组件
"""

import torch
import torch.nn as nn


class ECA(nn.Module):
    """Efficient Channel Attention (ECA) Module"""
    def __init__(self, c1, k_size=3):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: [B, C, H, W]
        y = self.avg_pool(x)  # [B, C, 1, 1]
        y = y.squeeze(-1).transpose(-1, -2)  # [B, 1, C]
        y = self.conv(y).transpose(-1, -2).unsqueeze(-1)  # [B, C, 1, 1]
        y = self.sigmoid(y)
        return x * y.expand_as(x)


# 注册模块到 Ultralytics (如果在训练脚本中调用)
def register_custom_modules():
    from ultralytics.nn.tasks import attempt_load_one_weight
    # 这里可以添加更多自定义逻辑
    pass
