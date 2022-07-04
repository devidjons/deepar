import torch
from torch.utils.data import Dataset


class TS_Dataset(Dataset):
    def __init__(self, df, x_cols, y_col, sequence_length=40):
        self.y_col = y_col
        self.x_cols = x_cols
        self.df_data = df.loc[:,x_cols]
        self.targets = df.loc[:,y_col]
        self.sequence_length = sequence_length
        self.batches = self.get_batches()
        
    def get_batches(self, ):
        n_batches = self.df_data.shape[0]//(self.sequence_length+1)
        return [self.generate_batch_by_start(i*(self.sequence_length+1)) for i in range(n_batches)]
        
    def generate_batch_by_start(self, start):
        x = self.df_data.loc[start:(start + self.sequence_length),:].values
        y = self.targets.loc[(start+1):(start + self.sequence_length+1)].values
        return (torch.tensor(x).float(), torch.tensor(y).reshape((-1,1)).float())

    def __len__(self):
        return len(self.batches)

    def __getitem__(self, idx):
        return self.batches[idx]