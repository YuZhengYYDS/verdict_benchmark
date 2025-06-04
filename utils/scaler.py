import numpy as np
import pickle
import os

class MinMaxScaler:
    def __init__(self, feature_range=(0, 1)):
        self.min = None
        self.max = None
        self.feature_range = feature_range

    def fit(self, data):
        data = self._ensure_numpy(data)
        self.min = np.min(data, axis=0)
        self.max = np.max(data, axis=0)
        return self

    def transform(self, data):
        self._check_fitted()
        data = self._ensure_numpy(data)
        scale = (self.feature_range[1] - self.feature_range[0]) / (self.max - self.min + 1e-8)
        return (data - self.min) * scale + self.feature_range[0]

    def inverse_transform(self, data):
        self._check_fitted()
        data = self._ensure_numpy(data)
        scale = (self.feature_range[1] - self.feature_range[0]) / (self.max - self.min + 1e-8)
        return (data - self.feature_range[0]) / scale + self.min

    def save(self, filepath):
        # 自动创建目录
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)
            
    @staticmethod
    def load(filepath):
        with open(filepath, 'rb') as f:
            return pickle.load(f)

    def _check_fitted(self):
        if self.min is None or self.max is None:
            raise ValueError("This MinMaxScaler instance is not fitted yet. Call 'fit' with training data.")

    @staticmethod
    def _ensure_numpy(data):
        if isinstance(data, np.ndarray):
            return data
        elif hasattr(data, 'cpu') and hasattr(data, 'numpy'):  # torch.Tensor
            return data.cpu().numpy()
        else:
            raise TypeError("Input must be a numpy array or a torch tensor.")

    def __repr__(self):
        return f"MinMaxScaler(feature_range={self.feature_range}, fitted={self.min is not None})"
