import torch
from torch.utils.data import Dataset
import numpy as np

class TimeSeriesDataset(Dataset):
    def __init__(self, df, lookback=24, x_mean=None, x_std=None, y_mean=None, y_std=None):
        values = df.drop(columns=["target"]).values.astype(np.float32)
        targets = df["target"].values.astype(np.float32)

        if x_mean is None:
            x_mean = values.mean(axis=0)
            x_std = values.std(axis=0) + 1e-6
            y_mean = targets.mean()
            y_std = targets.std() + 1e-6

        values = (values - x_mean) / x_std
        targets = (targets - y_mean) / y_std

        self.x_mean, self.x_std = x_mean, x_std
        self.y_mean, self.y_std = y_mean, y_std

        self.X, self.y = [], []
        for i in range(lookback, len(values)):
            self.X.append(values[i - lookback:i])
            self.y.append(targets[i])
        self.X = torch.tensor(self.X)
        self.y = torch.tensor(self.y)

    def __len__(self): return len(self.y)
    def __getitem__(self, idx): return self.X[idx], self.y[idx]
