import torch
import einops
import torch.nn as nn
import torch.nn.functional as F

from .blocks import MLPMixer

model = MLPMixer(
    image_size=224,
    patch_size=16,
    tokens_mlp_dim=128,
    channels_mlp_dim=128,
    total_classes=2,
    total_channels=4,
    total_blocks=4
)
