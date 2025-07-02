import torch
import torch.nn as nn
from models.mlp import get_activation

class ResidualBlock(nn.Module):
    """简单残差块，输入输出维度相同。"""
    def __init__(self, dim, activation):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.act = get_activation(activation)
        self.fc2 = nn.Linear(dim, dim)

    def forward(self, x):
        identity = x
        out = self.act(self.fc1(x))
        out = self.fc2(out)
        return self.act(out + identity)

class ResidualMLPRegressor(nn.Module):
    """
    残差 MLP：在多个隐层之间使用残差连接。
    Params:
      input_dim (int): 输入维度
      output_dim (int): 输出维度
      hidden_dims (list of int): 隐层尺寸列表
      activation (str): 激活函数名称
    """
    def __init__(self, input_dim, output_dim, hidden_dims, activation):
        super().__init__()
        self.activation = get_activation(activation)
        self.input_fc = nn.Linear(input_dim, hidden_dims[0])
        # 构建残差块
        self.blocks = nn.ModuleList([
            ResidualBlock(hidden_dims[i], activation)
            for i in range(len(hidden_dims)-1)
        ])
        self.output_fc = nn.Linear(hidden_dims[-1], output_dim)

    def forward(self, x):
        x = self.activation(self.input_fc(x))
        for block in self.blocks:
            x = block(x)
        return self.output_fc(x)
