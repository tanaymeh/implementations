import torch
import einops
import torch.nn as nn
import torch.nn.functional as F

from .blocks import MLPMixer

model = MLPMixer(
    image_size=224
)