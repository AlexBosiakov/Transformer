import torch
import torch.nn as nn

def unwrap_model(model):
    return model.module if hasattr(model, 'module') else model

class FinetuningStrategy:
    def apply(self, model, epoch, total_epochs):
        raise NotImplementedError

class StandardStrategy(FinetuningStrategy):
    def apply(self, model, epoch, total_epochs):
        pass

class FrozenEncoderStrategy(FinetuningStrategy):
    def __init__(self):
        self.applied = False
    def apply(self, model, epoch, total_epochs):
        if not self.applied:
            base = unwrap_model(model)
            for param in base.distilbert.parameters():
                param.requires_grad = False
            self.applied = True

class ExtraLayersStrategy(FinetuningStrategy):
    def __init__(self, hidden_dim=512, dropout=0.3):
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.applied = False

    def apply(self, model, epoch, total_epochs):
        if not self.applied:
            base = unwrap_model(model)
            original = base.classifier
            num_labels = original.out_features
            device = next(base.parameters()).device  # получаем устройство модели
            new_classifier = nn.Sequential(
                nn.Linear(768, self.hidden_dim),
                nn.ReLU(),
                nn.Dropout(self.dropout),
                nn.Linear(self.hidden_dim, num_labels)
            ).to(device)  # переносим на то же устройство
            base.classifier = new_classifier
            self.applied = True

class GradualUnfreezeStrategy(FinetuningStrategy):
    def __init__(self, num_layers=6, freeze_epochs=2):
        self.num_layers = num_layers
        self.freeze_epochs = freeze_epochs
        self.initialized = False
    def apply(self, model, epoch, total_epochs):
        base = unwrap_model(model)
        if not self.initialized:
            for param in base.distilbert.parameters():
                param.requires_grad = False
            self.initialized = True
        if epoch >= self.freeze_epochs:
            layers = min(epoch - self.freeze_epochs + 1, self.num_layers)
            for i, layer in enumerate(base.distilbert.transformer.layer[-layers:]):
                for param in layer.parameters():
                    param.requires_grad = True