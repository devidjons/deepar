import torch
from torch.utils.data import Dataset


class TS_Dataset(Dataset):
    def __init__(self, df, x_cols, y_col, sequence_length=40):
        self.y_col = y_col
        self.x_cols = x_cols
        self.df_data = df.loc[:,x_cols]
        self.targets = df.loc[:,y_col]
        self.sequence_length = sequence_length

    def __len__(self):
        return len(self.df_data)//self.sequence_length

    def __getitem__(self, idx):
        idx = idx * self.sequence_length
        # print("index = ", idx)
        if (idx+self.sequence_length+1) > len(self.df_data):
            idx-= self.sequence_length
        indexes = list(range(idx, idx + self.sequence_length+1))
        data = self.df_data.iloc[indexes[:-1], :].values
        target = self.targets.iloc[indexes[1:]].values
        return torch.tensor(data).float(), torch.tensor(target).reshape((-1,1)).float()