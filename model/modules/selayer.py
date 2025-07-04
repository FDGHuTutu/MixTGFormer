import torch
import torch.nn as nn
import torch.nn.functional as F

class SELayer(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        
        
        self.fc1 = nn.Linear(in_channels, in_channels // reduction)
        self.fc2 = nn.Linear(in_channels // reduction, in_channels)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        
        x_squeezed = x.mean(dim=[1, 2], keepdim=True)  #  [B, 1, 1, C]
        
        # Excitation
        x_squeezed = x_squeezed.view(x_squeezed.size(0), -1)  # [B, C]
        x_fc = F.relu(self.fc1(x_squeezed))
        x_fc = self.fc2(x_fc)
        x_scaled = self.sigmoid(x_fc).view(x.size(0), 1, 1, -1)  #  [B, 1, 1, C]
        
        # Scale
        return x * x_scaled
