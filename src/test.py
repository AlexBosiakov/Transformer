import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score

from model import TransformerEncoder
from dataset import NewsDatasetSPM


def test_model(
    model_path,
    test_df,
    tokenizer,
    batch_size=32,
    max_len=256,
    device="cuda"
):

    test_dataset = NewsDatasetSPM(test_df, tokenizer, max_len=max_len)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    model = TransformerEncoder(
        vocab_size=tokenizer.vocab_size(),
        d_model=256,
        max_len=max_len,
        num_heads=12,
        ff_hidden_dim=512,
        num_layers=4,
        num_classes=8
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

    print("=== TEST RESULTS ===")
    print(f"Accuracy:   {acc:.4f}")
    print(f"F1-macro:   {f1:.4f}")

    return acc, f1
