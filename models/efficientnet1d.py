import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from models.mlp import get_activation


class SqueezeExcitation1D(nn.Module):
    """Squeeze-and-Excitation block for 1D signals."""
    def __init__(self, channels, reduction=4):
        super().__init__()
        reduced_channels = max(1, channels // reduction)
        self.squeeze = nn.AdaptiveAvgPool1d(1)
        self.excitation = nn.Sequential(
            nn.Linear(channels, reduced_channels, bias=False),
            nn.SiLU(inplace=True),
            nn.Linear(reduced_channels, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _ = x.size()
        y = self.squeeze(x).view(b, c)
        y = self.excitation(y).view(b, c, 1)
        return x * y


class DepthwiseSeparableConv1D(nn.Module):
    """Depthwise separable convolution for efficiency."""
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=False):
        super().__init__()
        self.depthwise = nn.Conv1d(
            in_channels, in_channels, kernel_size, stride, padding, 
            groups=in_channels, bias=bias
        )
        self.pointwise = nn.Conv1d(in_channels, out_channels, 1, bias=bias)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x


class InvertedResidual1D(nn.Module):
    """Inverted residual block with depthwise separable convolution."""
    def __init__(self, in_channels, out_channels, expand_ratio=6, kernel_size=3, 
                 stride=1, se_ratio=0.25, activation='swish', dropout=0.0):
        super().__init__()
        self.stride = stride
        self.use_residual = stride == 1 and in_channels == out_channels
        
        # Expansion phase
        expanded_channels = in_channels * expand_ratio
        self.expand_conv = None
        if expand_ratio != 1:
            self.expand_conv = nn.Sequential(
                nn.Conv1d(in_channels, expanded_channels, 1, bias=False),
                nn.BatchNorm1d(expanded_channels),
                get_activation(activation)
            )
        
        # Depthwise convolution
        padding = kernel_size // 2
        self.depthwise_conv = nn.Sequential(
            DepthwiseSeparableConv1D(
                expanded_channels, expanded_channels, kernel_size, stride, padding
            ),
            nn.BatchNorm1d(expanded_channels),
            get_activation(activation)
        )
        
        # Squeeze-and-Excitation
        self.se = SqueezeExcitation1D(expanded_channels, int(1/se_ratio)) if se_ratio > 0 else None
        
        # Output projection
        self.project_conv = nn.Sequential(
            nn.Conv1d(expanded_channels, out_channels, 1, bias=False),
            nn.BatchNorm1d(out_channels)
        )
        
        # Dropout for regularization
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None

    def forward(self, x):
        identity = x
        
        # Expansion
        if self.expand_conv is not None:
            x = self.expand_conv(x)
        
        # Depthwise
        x = self.depthwise_conv(x)
        
        # Squeeze-and-Excitation
        if self.se is not None:
            x = self.se(x)
        
        # Project
        x = self.project_conv(x)
        
        # Dropout
        if self.dropout is not None:
            x = self.dropout(x)
        
        # Residual connection
        if self.use_residual:
            x = x + identity
            
        return x


class EfficientNet1DRegressor(nn.Module):
    """
    EfficientNet-inspired 1D CNN for signal regression with few parameters.
    
    This model uses:
    - Depthwise separable convolutions for efficiency
    - Inverted residual blocks
    - Squeeze-and-Excitation attention
    - Progressive channel expansion
    - Multi-scale global pooling
    
    Args:
        input_dim (int): Input signal dimension
        output_dim (int): Output dimension
        base_channels (int): Base number of channels
        depth_multiplier (float): Depth scaling factor
        width_multiplier (float): Width scaling factor
        activation (str): Activation function
        dropout (float): Dropout rate
        se_ratio (float): SE reduction ratio
    """
    
    def __init__(self, input_dim, output_dim, base_channels=32, depth_multiplier=1.0,
                 width_multiplier=1.0, activation='swish', dropout=0.2, se_ratio=0.25):
        super().__init__()
        
        # Scale channels based on width multiplier
        def make_divisible(v, divisor=8):
            return max(divisor, int(v + divisor / 2) // divisor * divisor)
        
        # Input projection to convert 1D signal to "spatial" representation
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            get_activation(activation),
            nn.Unflatten(1, (1, input_dim))  # (batch, 1, input_dim)
        )
        
        # Stem convolution
        stem_channels = make_divisible(base_channels * width_multiplier)
        self.stem = nn.Sequential(
            nn.Conv1d(1, stem_channels, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm1d(stem_channels),
            get_activation(activation)
        )
        
        # Efficient blocks configuration
        # (expand_ratio, channels, repeats, stride, kernel_size)
        blocks_config = [
            (1, 16, 1, 1, 3),   # Stage 1
            (6, 24, 2, 2, 3),   # Stage 2  
            (6, 40, 2, 2, 5),   # Stage 3
            (6, 80, 3, 2, 3),   # Stage 4
            (6, 112, 3, 1, 5),  # Stage 5
            (6, 192, 4, 2, 5),  # Stage 6
        ]
        
        # Build blocks
        self.blocks = nn.ModuleList()
        in_channels = stem_channels
        
        for expand_ratio, channels, repeats, stride, kernel_size in blocks_config:
            out_channels = make_divisible(channels * width_multiplier)
            num_repeats = max(1, int(repeats * depth_multiplier))
            
            # First block with stride
            self.blocks.append(
                InvertedResidual1D(
                    in_channels, out_channels, expand_ratio, kernel_size, stride,
                    se_ratio, activation, dropout
                )
            )
            in_channels = out_channels
            
            # Remaining blocks with stride=1
            for _ in range(num_repeats - 1):
                self.blocks.append(
                    InvertedResidual1D(
                        in_channels, out_channels, expand_ratio, kernel_size, 1,
                        se_ratio, activation, dropout
                    )
                )
        
        # Head convolution
        head_channels = make_divisible(320 * width_multiplier)
        self.head = nn.Sequential(
            nn.Conv1d(in_channels, head_channels, 1, bias=False),
            nn.BatchNorm1d(head_channels),
            get_activation(activation)
        )
        
        # Global feature aggregation with multiple pooling strategies
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        self.global_max_pool = nn.AdaptiveMaxPool1d(1)
        
        # Final classifier
        final_channels = head_channels * 2  # Concatenate avg and max pooling
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(final_channels, head_channels // 4),
            get_activation(activation),
            nn.Dropout(dropout / 2),
            nn.Linear(head_channels // 4, output_dim)
        )
        
        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize weights using proper schemes."""
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        # Input projection
        x = self.input_proj(x)  # (batch, 1, input_dim)
        
        # Stem
        x = self.stem(x)
        
        # Efficient blocks
        for block in self.blocks:
            x = block(x)
        
        # Head
        x = self.head(x)
        
        # Global pooling
        avg_pool = self.global_avg_pool(x)  # (batch, channels, 1)
        max_pool = self.global_max_pool(x)  # (batch, channels, 1)
        x = torch.cat([avg_pool, max_pool], dim=1)  # (batch, 2*channels, 1)
        
        # Classification
        x = self.classifier(x)
        
        return x

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
