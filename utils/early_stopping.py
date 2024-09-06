import numpy as np
import torch
import json

from utils.alert import alert
from utils.beep import beep

class EarlyStopping:
    # Carica il file di configurazione
    with open('config/config.json') as f:
        config = json.load(f)

    def __init__(self, patience=config['training']['patience'], delta=config['training']['delta']):
        self.patience = patience
        self.delta = delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.best_epoch_output = ""

    def __call__(self, val_loss, model, optimizer, epoch):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, optimizer, epoch)

        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'\033[34mEarly stopping counter: {self.counter} / {self.patience}\033[0m\n')
            if self.counter >= self.patience:
                self.early_stop = True
                print(f'\033[31mEarly stopping triggered: Fine allenamento\033[0m\n')
                alert()
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, optimizer, epoch)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, optimizer, epoch):
        print(f'\033[32mValidation loss diminuita ({self.val_loss_min:.2f} --> {val_loss:.2f}), salvataggio nuovo modello migliore...\033[0m\n')
        beep()
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_loss': val_loss,
        }, 'models/best_model.pth')
        self.val_loss_min = val_loss


