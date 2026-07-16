import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from tqdm import tqdm
from sklearn.metrics import f1_score
import csv
import os
import Optuna1

def train_model(
    model,
    train_dataset,
    val_dataset,
    epochs,
    batch_size=32,
    lr=1e-4,
    device="cuda",
    model_name="experiment",
    dropout=0.1,
    weight_decay=0.01,
    trial=None
):
    model.to(device)

    csv_path = f"/kaggle/working/{model_name}.csv"

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "model_name",
            "epoch",
            "dropout",
            "weight_decay",
            "train_loss",
            "val_loss",
            "val_acc",
            "val_f1"
        ])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_val_acc = 0.0

    print(f"Training model '{model_name}' with dropout={dropout}, weight_decay={weight_decay}")

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")

        for batch in pbar:
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)

            mask = batch["mask"].to(device)
            mask = mask.unsqueeze(1).unsqueeze(2)

            optimizer.zero_grad()

            logits = model(input_ids, mask)
            loss = criterion(logits, labels)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            pbar.set_postfix({
                "loss": f"{total_loss / (total + 1e-9):.4f}",
                "acc": f"{correct / (total + 1e-9):.4f}"
            })

        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0

        val_preds_all = []
        val_labels_all = []

        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                labels = batch["labels"].to(device)

                mask = batch["mask"].to(device)
                mask = mask.unsqueeze(1).unsqueeze(2)

                logits = model(input_ids, mask)
                loss = criterion(logits, labels)

                val_loss += loss.item()

                preds = logits.argmax(dim=1)

                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)

                val_preds_all.extend(preds.cpu().tolist())
                val_labels_all.extend(labels.cpu().tolist())

        val_acc = val_correct / val_total
        val_f1_macro = f1_score(val_labels_all, val_preds_all, average="macro")

        with open(csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                model_name,
                epoch + 1,
                dropout,
                weight_decay,
                total_loss / len(train_loader),
                val_loss / len(val_loader),
                val_acc,
                val_f1_macro
            ])

        if trial is not None:
            trial.report(val_acc, epoch)
            if trial.should_prune():
                print(f"Trial pruned at epoch {epoch + 1}")
                raise optuna.TrialPruned()
        print(f"\nEpoch {epoch+1} finished:")
        print(f"Train loss: {total_loss/len(train_loader):.4f}, Train acc: {correct/total:.4f}")
        print(f"Val loss:   {val_loss/len(val_loader):.4f}, Val acc:   {val_acc:.4f}")
        print(f"Val F1-macro: {val_f1_macro:.4f}")
        print(f"Dropout: {dropout}, Weight decay: {weight_decay}\n")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            checkpoint_path = f"/kaggle/working/{model_name}_best.pt"
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Saved new best model to {checkpoint_path}")

    return best_val_acc