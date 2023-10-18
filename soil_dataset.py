import torch
from torch.utils.data import Dataset
import numpy as np


class SoilDataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = torch.tensor(y, dtype=torch.float32)
        self.path = "data/processed/8e09234d1e1696d5c65e715b39d56b55/nbs"
        self.allx = torch.zeros((len(self.y),12,3,3), dtype=torch.float32)
        for i,ax in enumerate(self.x):
            scene = int(ax[0])
            row = int(ax[1])
            column = int(ax[2])
            file_name = f"{scene}_{row}_{column}.npy"
            file_path = f"{self.path}/{file_name}"
            data = np.load(file_path)
            self.allx[i] = torch.tensor(data, dtype=torch.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.allx[idx], self.y[idx]
