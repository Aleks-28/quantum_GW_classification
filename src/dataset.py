import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torch.utils.data import random_split

class TrainDataset(Dataset):
    
    def __init__(self,df):
        self.df = df
        self.path = df['path'].values
        self.labels = df['target'].values
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        data = np.load(self.path[idx])
        signal = torch.from_numpy(data).to(dtype=torch.float32).unsqueeze(1)
        signal_reshaped = signal.permute(1,0)
        label = torch.tensor(self.labels[idx]).to(dtype=torch.long)
        return signal_reshaped,label
