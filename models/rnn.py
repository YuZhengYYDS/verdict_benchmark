import torch
import torch.nn as nn
from models.mlp import get_activation

class RNNRegressor(nn.Module):
    """
    基于 RNN 的回归器，适配平坦特征输入。
    将输入特征重塑为序列以便 RNN 处理。
    
    Params:
      input_dim (int): 特征维度
      output_dim (int): 输出维度
      hidden_dim (int): RNN 隐状态维度
      num_layers (int): RNN 层数
      rnn_type (str): 'RNN' | 'LSTM' | 'GRU'
      activation (str): 激活函数
      dropout (float): 层间 dropout
      seq_len (int): 将输入重塑的序列长度，默认自动计算
    """
    def __init__(self, input_dim, output_dim, hidden_dim, num_layers, 
                 rnn_type='LSTM', activation='relu', dropout=0.0, seq_len=None):
        super().__init__()
        
        # Calculate optimal sequence length for reshaping
        if seq_len is None:
            # Find a good divisor of input_dim for reshaping
            divisors = [i for i in range(1, int(input_dim**0.5) + 1) if input_dim % i == 0]
            # Choose a reasonable sequence length (not too short, not too long)
            seq_len = min([d for d in divisors if d >= 8] + [divisors[-1]])
        
        self.seq_len = seq_len
        self.feature_dim = input_dim // seq_len
        
        # If input_dim is not divisible by seq_len, we need to pad or use a linear layer
        if input_dim % seq_len != 0:
            # Use a linear layer to project to a compatible size
            compatible_size = seq_len * hidden_dim
            self.input_projection = nn.Linear(input_dim, compatible_size)
            self.feature_dim = hidden_dim
            self.use_projection = True
        else:
            self.input_projection = None
            self.use_projection = False
        
        rnn_cls = {'RNN': nn.RNN, 'LSTM': nn.LSTM, 'GRU': nn.GRU}[rnn_type]
        self.rnn = rnn_cls(self.feature_dim, hidden_dim, num_layers,
                           batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.act = get_activation(activation)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # x: (batch, input_dim)
        batch_size = x.size(0)
        
        if self.use_projection:
            # Project to compatible size and reshape
            x = self.input_projection(x)  # (batch, seq_len * feature_dim)
            x = x.view(batch_size, self.seq_len, self.feature_dim)
        else:
            # Directly reshape
            x = x.view(batch_size, self.seq_len, self.feature_dim)
        
        # x is now: (batch, seq_len, feature_dim)
        out, _ = self.rnn(x)
        out = out[:, -1, :]  # Take the last timestep output
        out = self.act(out)
        return self.fc(out)