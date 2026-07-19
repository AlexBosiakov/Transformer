from transformers import Trainer, TrainingArguments
from sklearn.metrics import accuracy_score, f1_score

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = logits.argmax(axis=-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1_macro": f1_score(labels, preds, average="macro")
    }

class CustomTrainer(Trainer):
    def __init__(self, strategy=None, **kwargs):
        super().__init__(**kwargs)
        self.strategy = strategy

    def training_step(self, model, inputs):
        if self.strategy and self.state.epoch is not None:
            self.strategy.apply(model, int(self.state.epoch), self.args.num_train_epochs)
        return super().training_step(model, inputs)

def train_model(
    model,
    strategy,
    train_dataset,
    val_dataset,
    lr=2e-5,
    weight_decay=0.01,
    epochs=5,
    batch_size=32,
    output_dir="./hf_results",
    model_name="distilbert"
):
    training_args = TrainingArguments(
        output_dir=output_dir,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=lr,
        weight_decay=weight_decay,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        report_to=None,
        save_total_limit=2,
        logging_dir="./logs",
    )
    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        strategy=strategy
    )
    trainer.train()

    best_model_path = os.path.join(output_dir, f"best_{model_name}_{strategy}.pt")
    trainer.save_model(best_model_path)
    print(f"Best model saved to {best_model_path}")

    return trainer