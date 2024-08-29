import numpy as np
import torch

from utils.alert import alert
from utils.beep import beep

class EarlyStopping:
    def __init__(self, patience=5, delta=0):
        self.patience = patience
        self.delta = delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.best_epoch_output = ""  # Aggiungi questa variabile per memorizzare l'output dell'epoca migliore

    def __call__(self, val_loss, model, current_epoch_output=None):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, current_epoch_output)  # Passa l'output dell'epoca

        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'\033[34mEarly stopping counter: {self.counter} / {self.patience}\033[0m\n')
            if self.counter >= self.patience:
                self.early_stop = True
                print(f'\033[31mEarly stopping triggered: Fine allenamento\033[0m\n')
                alert()
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, current_epoch_output)  # Passa l'output dell'epoca
            self.counter = 0

    def save_checkpoint(self, val_loss, model, current_epoch_output):
        print(f'\033[32mValidation loss diminuita ({self.val_loss_min:.2f} --> {val_loss:.2f}), salvataggio nuovo modello migliore...\033[0m\n')
        beep()
        torch.save(model.state_dict(), 'models/best_model.pth')
        self.val_loss_min = val_loss
        self.best_epoch_output = current_epoch_output  # Memorizza l'output dell'epoca migliore
