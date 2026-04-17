import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
import numpy as np
import pandas as pd
from pathlib import Path
import os
from sklearn.metrics import classification_report, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
import json

class ADClassificationModel(pl.LightningModule):
    def __init__(self, model, learning_rate=1e-3, class_weights=None):
        super().__init__()
        self.model = model
        self.learning_rate = learning_rate
        self.criterion = nn.CrossEntropyLoss(weight=class_weights)

        # For tracking metrics
        self.train_losses = []
        self.val_losses = []
        self.val_accs = []
        self.epoch_metrics = []

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        self.log('train_loss', loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        acc = (y_hat.argmax(dim=1) == y).float().mean()
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', acc, prog_bar=True)
        self.val_losses.append(loss.item())
        self.val_accs.append(acc.item())
        return loss

    def on_validation_epoch_end(self):
        if self.val_losses and self.val_accs:
            avg_val_loss = np.mean(self.val_losses[-len(self.val_accs):])
            avg_val_acc = np.mean(self.val_accs[-len(self.val_accs):])
            avg_train_loss = np.mean(self.train_losses[-100:]) if self.train_losses else 0
            self.epoch_metrics.append({
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
                'val_acc': avg_val_acc
            })

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        acc = (y_hat.argmax(dim=1) == y).float().mean()
        self.log('test_loss', loss, prog_bar=True)
        self.log('test_acc', acc, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', patience=3, factor=0.5
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss"
            }
        }

def train_model(data_path, model_save_path, epochs=10, batch_size=32):
    """
    Train the 2D CNN model for AD classification.
    """
    # Load data splits
    splits_file = os.path.join(data_path, "data_splits.json")
    if not os.path.exists(splits_file):
        raise FileNotFoundError("Data splits not found. Run preprocess.py first.")

    with open(splits_file, 'r') as f:
        splits = json.load(f)

    from preprocess import get_data_loaders
    train_loader, val_loader, test_loader = get_data_loaders(splits, batch_size=batch_size)

    # Compute class weights to handle imbalance
    train_labels = splits['train']['labels']
    class_counts = np.bincount(train_labels, minlength=3).astype(float)
    class_weights = torch.tensor(1.0 / (class_counts / class_counts.sum()), dtype=torch.float32)
    class_weights = class_weights / class_weights.sum() * 3  # normalise to sum to num_classes

    # Create model
    from model import create_model
    model = create_model(num_classes=3)
    pl_model = ADClassificationModel(model, class_weights=class_weights)

    # Callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=model_save_path,
        filename='ad-classifier-{epoch:02d}-{val_acc:.2f}',
        save_top_k=3,
        monitor='val_acc',
        mode='max',
        save_weights_only=True
    )

    early_stop_callback = EarlyStopping(
        monitor='val_loss',
        patience=5,
        mode='min'
    )

    # Trainer
    trainer = pl.Trainer(
        max_epochs=epochs,
        callbacks=[checkpoint_callback, early_stop_callback],
        accelerator='mps' if torch.backends.mps.is_available() else ('gpu' if torch.cuda.is_available() else 'cpu'),
        devices=1,
        log_every_n_steps=100,
        enable_progress_bar=True
    )

    # Train
    trainer.fit(pl_model, train_loader, val_loader)

    # Test
    test_results = trainer.test(pl_model, test_loader)

    print(f"Training completed. Best model saved at: {checkpoint_callback.best_model_path}")

    # Save training history
    history = {
        'best_model_path': checkpoint_callback.best_model_path,
        'best_val_acc': checkpoint_callback.best_model_score.item() if checkpoint_callback.best_model_score else None,
        'test_results': test_results,
        'epoch_metrics': pl_model.epoch_metrics
    }

    with open(os.path.join(model_save_path, 'training_history.json'), 'w') as f:
        json.dump(history, f, indent=2)

    return checkpoint_callback.best_model_path

if __name__ == "__main__":
    data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed')
    model_save_path = os.path.join(os.path.dirname(__file__), '..', 'models')

    os.makedirs(model_save_path, exist_ok=True)

    best_model_path = train_model(data_path, model_save_path, epochs=5)  # Reduced epochs for demo