import scipy.io
import numpy as np
from torch.utils.data import Dataset, DataLoader, random_split
from utils.scaler import MinMaxScaler
import torch
import os

class VerdictDataset(Dataset):
    def __init__(self, mat_path, scaler_path=None, fit_scaler=True):
        mat = scipy.io.loadmat(mat_path)
        self.X = mat['database_train_noisy'].astype(np.float32)
        self.y = mat['params_train'].astype(np.float32)

        # 是否加载现有Scaler
        if scaler_path and os.path.exists(scaler_path):
            self.scaler = MinMaxScaler.load(scaler_path)
            self.y = self.scaler.transform(self.y)
        else:
            self.scaler = MinMaxScaler()
            if fit_scaler:
                self.scaler.fit(self.y)
                self.y = self.scaler.transform(self.y)
            if scaler_path:
                self.scaler.save(scaler_path)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
    
def get_dataloaders(mat_path, batch_size, train_ratio=0.8, seed=42, scaler_path=None):
    full_dataset = VerdictDataset(mat_path, scaler_path=scaler_path, fit_scaler=True)
    total = len(full_dataset)
    train_len = int(train_ratio * total)
    val_len = total - train_len
    torch_gen = torch.Generator().manual_seed(seed)
    train_set, val_set = random_split(full_dataset, [train_len, val_len], generator=torch_gen)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader
