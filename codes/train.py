import torch
import time
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import os

from codes.evaluate import evaluate
from utils.early_stopping import EarlyStopping
from utils.console_output_manager import suppress_stdout, enable_stdout
from utils.save_epoch_out import save_epoch_output
from utils.time_manager import get_current_time, calculate_epoch_time, format_total_time

# Definizione della funzione load_checkpoint
def load_checkpoint(model, optimizer, filepath):
    checkpoint = torch.load(filepath, weights_only=True)
    
    # Carica solo le chiavi che corrispondono all'architettura del modello
    model_state_dict = model.state_dict()
    checkpoint_state_dict = checkpoint['model_state_dict']
    
    # Filtra i pesi del checkpoint per mantenere solo quelli che il modello si aspetta
    filtered_state_dict = {k: v for k, v in checkpoint_state_dict.items() if k in model_state_dict}
    
    model_state_dict.update(filtered_state_dict)
    model.load_state_dict(model_state_dict)
    
    if 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    start_epoch = checkpoint.get('epoch', 0)
    return start_epoch

# Definizione della funzione train
def train(model, train_loader, val_loader, criterion, optimizer, scheduler, config, start_epoch=0):
    writer = SummaryWriter('runs')
    epochs = config['training']['epochs']
    early_stopping = EarlyStopping(patience=config['training']['patience'], delta=config['training']['delta'])
    sum_time = 0.0
    last_epoch_output = ""

    try:
        for epoch in range(start_epoch, epochs):
            start_time = time.time()

            # Inizia addestramento
            model.train()
            running_loss = 0.0
            correct = 0
            total = 0

            print("----------------------------------------------------------------------------------------------")

            # Itera sul dataset di training
            for i, (inputs, labels) in enumerate(tqdm(train_loader, desc='Training...', ncols=100)):
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

            train_loss = running_loss / len(train_loader)
            train_accuracy = 100 * correct / total

            val_loss, val_accuracy = evaluate(model, val_loader, criterion)

            writer.add_scalar('Loss/train', train_loss, epoch)
            writer.add_scalar('Accuracy/train', train_accuracy, epoch)
            writer.add_scalar('Loss/validation', val_loss, epoch)
            writer.add_scalar('Accuracy/validation', val_accuracy, epoch)

            epoch_time = calculate_epoch_time(start_time)
            sum_time += epoch_time

            current_time = get_current_time()

            last_epoch_output = (
                f"Epoca [{epoch + 1}/{epochs}]: "
                f"\n- Train Loss: {train_loss:.2f}"
                f"\n- Train Accuracy: {train_accuracy:.2f}%"
                f"\n- Validation Loss: \033[33m{val_loss:.2f}\033[0m"
                f"\n- Validation Accuracy: \033[33m{val_accuracy:.2f}%\033[0m"
                f"\n\033[34m[ \033[36m{current_time} \033[34m- Tempo impiegato: {epoch_time:.2f}s ]\033[0m"
            )
            print(last_epoch_output)

            suppress_stdout()
            enable_stdout()

            scheduler.step(val_loss)

            # Passaggio dell'argomento epoch
            early_stopping(val_loss, model, optimizer, epoch)

            if early_stopping.early_stop:
                print("\033[31mEarly stopping attivato\033[0m")
                break
    
    except KeyboardInterrupt:
        print("\033[31mTraining interrotto manualmente\033[0m")
    
    finally:
        total_time_formatted = format_total_time(sum_time)
        print(f'\n\033[34m[ Tempo totale impiegato: {total_time_formatted} ]\033[0m')
        writer.close()

        if early_stopping.best_epoch_output:
            save_epoch_output(early_stopping.best_epoch_output)
