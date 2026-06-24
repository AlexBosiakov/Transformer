import torch
from torch.utils.data import Dataset

class NewsDatasetSPM(Dataset):
    def __init__(self, df, tokenizer, max_len=256):
        self.df = df
        self.tokenizer = tokenizer
        self.max_len = max_len

        self.cls_id = tokenizer.bos_id() if tokenizer.bos_id() != -1 else 1

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        text = row["Title"] + " " + row["Description"]
        label = int(row["Class Index"]) - 1

        # токенизация
        ids = self.tokenizer.encode(text, out_type=int)

        # обрезаем (оставляем место под CLS)
        if len(ids) > self.max_len - 1:
            ids = ids[: self.max_len - 1]

        # добавляем CLS в начало
        ids = [self.cls_id] + ids

        # паддинг
        pad_len = self.max_len - len(ids)
        ids = ids + [0] * pad_len

        # attention mask
        mask = [1 if token_id != 0 else 0 for token_id in ids]

        return {
            "input_ids": torch.tensor(ids, dtype=torch.long),
            "mask": torch.tensor(mask, dtype=torch.long),
            "labels": torch.tensor(label, dtype=torch.long)
        }