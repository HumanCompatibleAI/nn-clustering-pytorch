import numpy as np
import torch
from torch.utils.data import Dataset

# tiny dataset meant for debugging

# randomly generated xs
xs = np.array([[0.06320405, 0.51371515, 0.04077784],
               [0.58809062, 0.58997539, 0.31045666],
               [0.22153995, 0.81825784, 0.31460745],
               [0.37792006, 0.05979807, 0.35770925],
               [0.83125488, 0.50243196, 0.57578912],
               [0.52504822, 0.57349545, 0.00399584],
               [0.78231748, 0.21112105, 0.92726576],
               [0.77467497, 0.76202789, 0.63063908],
               [0.84998826, 0.00393768, 0.64844679],
               [0.18798294, 0.54178095, 0.60813651]])

ys = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]


class TinyDataset(Dataset):
    def __init__(self):
        self.xs = xs
        self.ys = ys

    def __len__(self):
        return len(self.ys)

    def __getitem__(self, idx):
        x_np_array = xs[idx]
        x_tens = torch.from_numpy(x_np_array).float()
        y = ys[idx]
        y_tens = torch.tensor(y)
        return x_tens, y_tens
