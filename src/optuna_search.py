import optuna
from transformers import Trainer, TrainingArguments
from hf_model import create_model
from hf_train import compute_metrics


def run_optuna(train_dataset, val_dataset, model_name="distilbert-base-uncased", n_trials=20):
    def objective(trial):
        lr = trial.suggest_float("lr", 1e-5, 5e-4, log=True)
        wd = trial.suggest_float("weight_decay", 1e-5, 0.1, log=True)
        warmup_ratio = trial.suggest_float("warmup_ratio", 0.0, 0.2)
        batch_size = trial.suggest_categorical("batch_size", [16, 32])

        model, _ = create_model("standard", model_name=model_name)
        args = TrainingArguments(
            output_dir="./optuna",
            learning_rate=lr,
            weight_decay=wd,
            warmup_ratio=warmup_ratio,
            num_train_epochs=3,
            evaluation_strategy="epoch",
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            report_to=None,
            save_strategy="no"
        )
        trainer = Trainer(
            model=model,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=compute_metrics
        )
        trainer.train()
        return trainer.evaluate()["eval_accuracy"]

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)
    return study.best_params