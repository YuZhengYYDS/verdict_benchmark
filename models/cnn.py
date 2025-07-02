import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from models.mlp import get_activation


class ChannelAttention(nn.Module):
    """Channel attention mechanism for signal features."""
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction, bias=False),
            nn.ReLU(),
            nn.Linear(in_channels // reduction, in_channels, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _ = x.size()
        avg_out = self.fc(self.avg_pool(x).view(b, c))
        max_out = self.fc(self.max_pool(x).view(b, c))
        out = avg_out + max_out
        return self.sigmoid(out).view(b, c, 1)


class SpatialAttention(nn.Module):
    """Spatial attention mechanism for temporal features."""
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv1d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_cat = torch.cat([avg_out, max_out], dim=1)
        x_cat = self.conv(x_cat)
        return self.sigmoid(x_cat)


class MultiScaleConvBlock(nn.Module):
    """Multi-scale convolution block with different kernel sizes."""
    def __init__(self, in_channels, out_channels, activation='relu', dropout=0.0):
        super().__init__()
        
        # Different scales for feature extraction
        self.conv_small = nn.Conv1d(in_channels, out_channels//4, 3, padding=1)
        self.conv_medium = nn.Conv1d(in_channels, out_channels//4, 5, padding=2)
        self.conv_large = nn.Conv1d(in_channels, out_channels//4, 7, padding=3)
        self.conv_xlarge = nn.Conv1d(in_channels, out_channels//4, 11, padding=5)
        
        self.bn = nn.BatchNorm1d(out_channels)
        self.activation = get_activation(activation)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        
        # Channel attention
        self.ca = ChannelAttention(out_channels)
        # Spatial attention
        self.sa = SpatialAttention()

    def forward(self, x):
        # Multi-scale feature extraction
        x1 = self.conv_small(x)
        x2 = self.conv_medium(x)
        x3 = self.conv_large(x)
        x4 = self.conv_xlarge(x)
        
        # Concatenate multi-scale features
        x = torch.cat([x1, x2, x3, x4], dim=1)
        x = self.bn(x)
        x = self.activation(x)
        
        # Apply attention mechanisms
        x = x * self.ca(x)
        x = x * self.sa(x)
        
        x = self.dropout(x)
        return x


class ResidualBlock(nn.Module):
    """Residual block for better gradient flow."""
    def __init__(self, channels, kernel_size=3, activation='relu', dropout=0.0):
        super().__init__()
        self.conv1 = nn.Conv1d(channels, channels, kernel_size, padding=kernel_size//2)
        self.bn1 = nn.BatchNorm1d(channels)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size, padding=kernel_size//2)
        self.bn2 = nn.BatchNorm1d(channels)
        self.activation = get_activation(activation)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.activation(out)
        out = self.dropout(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual  # Residual connection
        out = self.activation(out)
        return out


class CNNRegressor(nn.Module):
    """
    Advanced 1D CNN for signal regression with multi-scale feature extraction,
    attention mechanisms, and residual connections.
    
    This CNN is designed specifically for meaningful signal processing:
    - Multi-scale convolutions capture features at different temporal scales
    - Attention mechanisms focus on important signal regions
    - Residual connections enable deeper networks
    - Adaptive pooling for flexible input sizes
    
    Params:
      input_dim (int): Signal length (number of features)
      output_dim (int): Output dimension
      base_filters (int): Base number of filters (default: 32)
      num_blocks (int): Number of multi-scale blocks (default: 3)
      activation (str): Activation function name
      dropout (float): Dropout probability
      use_residual (bool): Whether to use residual connections
    """
    def __init__(self, input_dim, output_dim, base_filters=32, num_blocks=3, 
                 activation='relu', dropout=0.1, use_residual=True):
        super().__init__()
        
        self.input_dim = input_dim
        self.use_residual = use_residual
        
        # Initial embedding layer
        self.embedding = nn.Sequential(
            nn.Conv1d(1, base_filters, 7, padding=3),
            nn.BatchNorm1d(base_filters),
            get_activation(activation)
        )
        
        # Multi-scale feature extraction blocks
        self.feature_blocks = nn.ModuleList()
        in_channels = base_filters
        
        for i in range(num_blocks):
            out_channels = base_filters * (2 ** i)
            self.feature_blocks.append(
                MultiScaleConvBlock(in_channels, out_channels, activation, dropout)
            )
            
            # Add residual blocks for deeper feature learning
            if use_residual:
                self.feature_blocks.append(
                    ResidualBlock(out_channels, activation=activation, dropout=dropout)
                )
            
            # Adaptive pooling to reduce temporal dimension
            if i < num_blocks - 1:  # No pooling on last block
                self.feature_blocks.append(nn.AdaptiveAvgPool1d(input_dim // (2 ** (i + 1))))
            
            in_channels = out_channels
        
        # Global feature aggregation
        final_channels = base_filters * (2 ** (num_blocks - 1))
        
        # Multiple pooling strategies for robust feature extraction
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        self.global_max_pool = nn.AdaptiveMaxPool1d(1)
        
        # Feature fusion and final prediction
        self.classifier = nn.Sequential(
            nn.Linear(final_channels * 2, final_channels),  # *2 for avg+max pooling
            nn.BatchNorm1d(final_channels),
            get_activation(activation),
            nn.Dropout(dropout),
            nn.Linear(final_channels, final_channels // 2),
            nn.BatchNorm1d(final_channels // 2),
            get_activation(activation),
            nn.Dropout(dropout),
            nn.Linear(final_channels // 2, output_dim)
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize network weights."""
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # Input: (batch_size, input_dim)
        x = x.unsqueeze(1)  # (batch_size, 1, input_dim)
        
        # Initial embedding
        x = self.embedding(x)  # (batch_size, base_filters, input_dim)
        
        # Multi-scale feature extraction
        for block in self.feature_blocks:
            x = block(x)
        
        # Global feature aggregation
        avg_pool = self.global_avg_pool(x).squeeze(-1)  # (batch_size, channels)
        max_pool = self.global_max_pool(x).squeeze(-1)  # (batch_size, channels)
        
        # Concatenate different pooling results
        x = torch.cat([avg_pool, max_pool], dim=1)  # (batch_size, channels*2)
        
        # Final prediction
        x = self.classifier(x)
        
        return x