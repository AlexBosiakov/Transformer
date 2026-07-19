import torch
from torch.utils.data import Dataset

class HFDataset(Dataset):
    def __init__(self, df, tokenizer, max_len=128, text_columns=["Title", "Description"]):
        self.df = df
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.text_columns = text_columns

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        # Объединяем заголовок и описание
        text = row["Title"] + " " + row["Description"]
        label = int(row["Class Index"]) - 1  # классы 1..4 → 0..3

        encoding = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt"
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(label, dtype=torch.long)
        }