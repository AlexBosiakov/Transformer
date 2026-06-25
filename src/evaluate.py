import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score
import csv
import os

from model import TransformerEncoder
from dataset import NewsDatasetSPM


def test_model(
    model_path,
    test_df,
    tokenizer,
    *,
    batch_size,
    max_len,
    device,
    d_model,
    num_heads,
    ff_hidden_dim,
    num_layers,
    num_classes,
    dropout=0.1,
    weight_decay=0.01
):

    test_dataset = NewsDatasetSPM(test_df, tokenizer, max_len=max_len)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    model = TransformerEncoder(
        vocab_size=tokenizer.vocab_size(),
        d_model=d_model,
        max_len=max_len,
        num_heads=num_heads,
        ff_hidden_dim=ff_hidden_dim,
        num_layers=num_layers,
        num_classes=num_classes,
        dropout=dropout
    )

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    preds_all = []
    labels_all = []

    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)

            mask = batch["mask"].to(device)
            mask = mask.unsqueeze(1).unsqueeze(2)

            logits = model(input_ids, mask)
            preds = logits.argmax(dim=1)

            preds_all.extend(preds.cpu().tolist())
            labels_all.extend(labels.cpu().tolist())

    acc = accuracy_score(labels_all, preds_all)
    f1 = f1_score(labels_all, preds_all, average="macro")

    print("TEST RESULTS")
    print(f"Accuracy:     {acc:.4f}")
    print(f"F1-macro:     {f1:.4f}")
    print(f"Dropout:      {dropout}")
    print(f"Weight decay: {weight_decay}")

    csv_path = "/kaggle/working/test_results.csv"
    file_exists = os.path.isfile(csv_path)

    with open(csv_path, "a", newline="") as f:
        writer = csv.writer(f)

        if not file_exists:
            writer.writerow([
                "model_path",
                "d_model",
                "num_heads",
                "ff_hidden_dim",
                "num_layers",
                "num_classes",
                "batch_size",
                "max_len",
                "dropout",
                "weight_decay",
                "accuracy",
                "f1_macro"
            ])

        writer.writerow([
            model_path,
            d_model,
            num_heads,
            ff_hidden_dim,
            num_layers,
            num_classes,
            batch_size,
            max_len,
            dropout,
            weight_decay,
            acc,
            f1
        ])

    return acc, f1
