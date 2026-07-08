import Optuna1
from model import TransformerEncoder
import sentencepiece as spm
from train import train_model
from dataset import NewsDatasetSPM

def objective(trial):
    # 1. Предлагаем гиперпараметры
    d_model = trial.suggest_int("d_model", 128, 512, step=64)
    head_dim = trial.suggest_int("head_dim", 32, 64, step=16)
    num_heads = d_model // head_dim
    if num_heads < 2 or num_heads > 16:
        raise optuna.TrialPruned()

    ff_hidden_dim = trial.suggest_int("ff_hidden_dim", d_model*2, d_model*4, step=64)
    num_layers = trial.suggest_int("num_layers", 2, 6)
    dropout = trial.suggest_float("dropout", 0.1, 0.5, step=0.1)
    lr = trial.suggest_float("lr", 5e-5, 5e-4, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-5, 3.0, log=True)

    sp = spm.SentencePieceProcessor()
    sp.load("spm.model")

    # 2. Создаём модель
    model = TransformerEncoder(
        vocab_size=sp.vocab_size(),
        d_model=d_model,
        max_len=256,
        num_heads=num_heads,
        ff_hidden_dim=ff_hidden_dim,
        num_layers=num_layers,
        num_classes=4,
        dropout=dropout
    )

    model_name = f"optuna_trial_{trial.number}"

    # 3. Запускаем обучение (передаём trial для pruning)
    best_val_acc = train_model(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        epochs=10,
        batch_size=32,
        lr=lr,
        device="cuda",
        model_name=model_name,
        dropout=dropout,
        weight_decay=weight_decay,
        trial=trial
    )

    return best_val_acc