# src/verdict_benchmark/data/loader.py

import os
import numpy as np
import scipy.io
from torch.utils.data import Dataset


class VerdictDataset(Dataset):
    """
    PyTorch Dataset: 给定 signals.npy（或从 .mat 加载）和 params.npy，输出 (signal_tensor, param_tensor)
    """

    def __init__(self, signals_path, params_path, transform=None):
        """
        signals_path: npy 文件或 .mat 中的键
        params_path: npy 文件或 .mat 中的键
        """
        # 如果是 .npy 直接加载
        if signals_path.endswith('.npy'):
            self.signals = np.load(signals_path)  # shape: (N_samples, N_features)
        elif signals_path.endswith('.mat'):
            mat = scipy.io.loadmat(signals_path)
            # 假设 .mat 中保存信号的变量名叫 'database_train_noisy' 或 'database_test_noisy'
            key = 'database_train_noisy' if 'train' in os.path.basename(signals_path) else 'database_test_noisy'
            self.signals = mat[key].astype(np.float32)
        else:
            raise ValueError("signals_path 必须是 .npy 或 .mat 文件")

        if params_path.endswith('.npy'):
            self.params = np.load(params_path)  # shape: (N_samples, N_targets)
        elif params_path.endswith('.mat'):
            mat = scipy.io.loadmat(params_path)
            key = 'params_train' if 'train' in os.path.basename(params_path) else 'params_test'
            self.params = mat[key].astype(np.float32)
        else:
            raise ValueError("params_path 必须是 .npy 或 .mat 文件")

        assert self.signals.shape[0] == self.params.shape[0], "样本数量不匹配"

        self.transform = transform

    def __len__(self):
        return self.signals.shape[0]

    def __getitem__(self, idx):
        x = self.signals[idx]    # NumPy array, 一维：N_features
        y = self.params[idx]     # NumPy array, 一维：N_targets

        if self.transform:
            x = self.transform(x)

        # 转为 PyTorch tensor
        import torch
        x = torch.from_numpy(x).float()
        y = torch.from_numpy(y).float()

        return x, y
