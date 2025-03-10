import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class CSLData(Dataset):
    def __init__(self, npz_path):
        data = np.load(npz_path)
        self.skeletal = torch.FloatTensor(data["skeletal_data"])
        self.crops = torch.FloatTensor(data["crops"])
        self.flow = torch.FloatTensor(data["optical_flow"])
        self.labels = data["labels"]

    def __len__(self):
        return 1  # Single video for now; expand for multiple

    def __getitem__(self, idx):
        return self.skeletal, self.crops, self.flow, self.labels

dataset = CSLData("preprocessed.npz")
loader = DataLoader(dataset, batch_size=1)