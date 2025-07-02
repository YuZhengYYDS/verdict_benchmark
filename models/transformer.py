import torch
import torch.nn as nn
from models.mlp import get_activation

class TransformerRegressor(nn.Module):
    """
    基于 Transformer Encoder 的回归器。
    Params:
      input_dim (int): 输入特征维度
      output_dim (int): 输出维度
      d_model (int): Transformer 隐特征维度
      nhead (int): 注意力头数
      num_layers (int): Encoder 层数
      dim_feedforward (int): 前馈网络维度
      activation (str): 'relu' or 'gelu'
      dropout (float): dropout 比例
    """
    def __init__(self, input_dim, output_dim, d_model=64, nhead=4,
                 num_layers=2, dim_feedforward=128,
                 activation='relu', dropout=0.1):
        super().__init__()
        self.input_fc = nn.Linear(input_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_fc = nn.Linear(d_model, output_dim)

    def forward(self, x):
        # x: (batch, input_dim)
        x = x.unsqueeze(1)                # (batch, seq_len=1, input_dim)
        x = self.input_fc(x)             # (batch, seq_len, d_model)
        x = self.transformer(x)          # (batch, seq_len, d_model)
        x = x.mean(dim=1)                # (batch, d_model)
        return self.output_fc(x)
