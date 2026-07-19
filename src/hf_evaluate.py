import torch
import csv
import os
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader


def evaluate_model(
        model,
        test_dataset,
        device="cuda",
        model_name="distilbert",
        strategy="standard",
        lr=2e-5,
        weight_decay=0.01,
        dropout=None,
        num_epochs=5
):
    model.to(device)
    model.eval()
    loader = DataLoader(test_dataset, batch_size=32)
    preds_all = []
    labels_all = []
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            outputs = model(input_ids, attention_mask=attention_mask)
            preds = outputs.logits.argmax(dim=1)
            preds_all.extend(preds.cpu().tolist())
            labels_all.extend(labels.cpu().tolist())
    acc = accuracy_score(labels_all, preds_all)
    f1 = f1_score(labels_all, preds_all, average="macro")

    # Сохранение в CSV (только меняющиеся параметры)
    if os.path.exists("/kaggle/working"):
        output_dir = "/kaggle/working"
    else:
        output_dir = "."
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, "hf_results.csv")
    file_exists = os.path.isfile(csv_path)

    with open(csv_path, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow([
                "model_name",
                "strategy",
                "learning_rate",
                "weight_decay",
                "dropout",
                "num_epochs",
                "accuracy",
                "f1_macro"
            ])
        writer.writerow([
            model_name,
            strategy,
            lr,
            weight_decay,
            dropout if dropout is not None else "N/A",
            num_epochs,
            acc,
            f1
        ])

    print(f"✅ Results saved to {csv_path}")
    return {"accuracy": acc, "f1_macro": f1}