from transformers import AutoModelForSequenceClassification
from strategies import (
    StandardStrategy, FrozenEncoderStrategy,
    ExtraLayersStrategy, GradualUnfreezeStrategy
)


def create_model(strategy_name, model_name="distilbert-base-uncased", num_labels=4, **kwargs):
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)

    if strategy_name == "standard":
        strategy = StandardStrategy()
    elif strategy_name == "frozen_encoder":
        strategy = FrozenEncoderStrategy()
    elif strategy_name == "extra_layers":
        strategy = ExtraLayersStrategy(
            hidden_dim=kwargs.get("hidden_dim", 512),
            dropout=kwargs.get("dropout", 0.3)
        )
    elif strategy_name == "gradual_unfreeze":
        strategy = GradualUnfreezeStrategy(
            num_layers=kwargs.get("num_layers", 6),
            freeze_epochs=kwargs.get("freeze_epochs", 2)
        )
    else:
        raise ValueError(f"Unknown strategy: {strategy_name}")

    return model, strategy