import os
import torch
from torch.utils.data import Dataset, DataLoader
import scipy.io
import numpy as np
import platform
from typing import Optional, Tuple
from utils.scaler import MinMaxScaler


def load_mat_data(mat_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load raw data arrays from a MATLAB .mat file.

    Returns:
        X (np.ndarray): Input features, shape (N, D).
        y (np.ndarray): Target labels, shape (N, K).
    """
    mat = scipy.io.loadmat(mat_path)
    X = mat['database_train_noisy'].astype(np.float32)
    y = mat['params_train'].astype(np.float32)
    return X, y


class VerdictDataset(Dataset):
    """
    Simple in-memory PyTorch Dataset for Verdict data.
    """
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = X
        self.y = y

    def __len__(self) -> int:
        return self.X.shape[0]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x = torch.from_numpy(self.X[idx])
        y = torch.from_numpy(self.y[idx])
        return x, y


def get_dataloaders(
    mat_path: str,
    batch_size: int,
    train_ratio: float = 0.8,
    seed: int = 42,
    scaler_path: Optional[str] = None,
    num_workers: int = None,
    pin_memory: bool = True
) -> Tuple[DataLoader, DataLoader]:
    """
    Create train/validation DataLoaders from a .mat dataset.

    - Loads raw numpy arrays once.
    - Splits by index permutation for reproducible splits.
    - Fits MinMaxScaler on y_train only, applies to val.
    - Supports saving/loading scaler parameters.
    - Uses num_workers and pin_memory for performance.
    - Automatically sets num_workers=0 on Windows to avoid multiprocessing issues.

    Returns:
        train_loader, val_loader
    """
    # Load raw arrays
    X, y = load_mat_data(mat_path)
    total = X.shape[0]

    # Reproducible random split
    generator = torch.Generator().manual_seed(seed)
    perm = torch.randperm(total, generator=generator)
    train_len = int(train_ratio * total)
    train_idx = perm[:train_len].numpy()
    val_idx = perm[train_len:].numpy()

    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]

    # Fit scaler on train targets only
    scaler = MinMaxScaler()
    scaler.fit(y_train)
    y_train = scaler.transform(y_train)
    y_val = scaler.transform(y_val)
    if scaler_path:
        scaler.save(scaler_path)

    # Create datasets
    train_dataset = VerdictDataset(X_train, y_train)
    val_dataset = VerdictDataset(X_val, y_val)

    # Set num_workers based on platform - Windows has multiprocessing issues
    if num_workers is None:
        num_workers = 0 if platform.system() == 'Windows' else 4

    # Create loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    return train_loader, val_loader
