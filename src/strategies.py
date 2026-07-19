import torch
import torch.nn as nn

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
            for param in model.distilbert.parameters():
                param.requires_grad = False
            self.applied = True

class ExtraLayersStrategy(FinetuningStrategy):
    def __init__(self, hidden_dim=512, dropout=0.3):
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.applied = False
    def apply(self, model, epoch, total_epochs):
        if not self.applied:
            original = model.classifier
            num_labels = original.out_features
            model.classifier = nn.Sequential(
                nn.Linear(768, self.hidden_dim),
                nn.ReLU(),
                nn.Dropout(self.dropout),
                nn.Linear(self.hidden_dim, num_labels)
            )
            self.applied = True

class GradualUnfreezeStrategy(FinetuningStrategy):
    def __init__(self, num_layers=6, freeze_epochs=2):
        self.num_layers = num_layers
        self.freeze_epochs = freeze_epochs
        self.initialized = False
    def apply(self, model, epoch, total_epochs):
        if not self.initialized:
            for param in model.distilbert.parameters():
                param.requires_grad = False
            self.initialized = True
        if epoch >= self.freeze_epochs:
            layers = min(epoch - self.freeze_epochs + 1, self.num_layers)
            for i, layer in enumerate(model.distilbert.transformer.layer[-layers:]):
                for param in layer.parameters():
                    param.requires_grad = True