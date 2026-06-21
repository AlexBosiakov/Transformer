import torch
from torch.utils.data import Dataset

class NewsDatasetSPM(Dataset):
    def __init__(self, df, tokenizer, max_len=256):
        self.df = df
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        text = row["Title"] + " " + row["Description"]
        label = int(row["Class Index"]) - 1

        input_ids = self.tokenizer.encode(text, max_len=self.max_len)

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(label, dtype=torch.long)
        }