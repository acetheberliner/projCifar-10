import torch
import sys
import os
import time

from torch.utils.tensorboard import SummaryWriter
from utils.early_stopping import EarlyStopping
from codes.evaluate import evaluate

from utils.console_output_manager import suppress_stdout, enable_stdout
from utils.save_epoch_out import save_epoch_output

def train(model, train_loader, val_loader, criterion, optimizer, scheduler, config):
    writer = SummaryWriter('runs')
    epochs = config['training']['epochs']
    early_stopping = EarlyStopping(patience=config['training']['patience'], delta=config['training']['delta'])
    sum_time = 0.0
    target_accuracy = 90.0
    last_epoch_output = ""

    try:
        for epoch in range(epochs):
            start_time = time.time()

            model.train()
            running_loss = 0.0
            correct = 0
            total = 0

            for i, (inputs, labels) in enumerate(train_loader):
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

            epoch_time = time.time() - start_time
            sum_time += epoch_time

            current_time = time.strftime('%H:%M:%S', time.localtime())

            last_epoch_output = (
                "----------------------------------------------------------------------------------------------\n"
                f"Epoca [{epoch + 1}/{epochs}]: "
                f"\n- Train Loss: {train_loss:.2f}"
                f"\n- Train Accuracy: {train_accuracy:.2f}%"
                f"\n- Validation Loss: \033[33m{val_loss:.2f}\033[0m"
                f"\n- Validation Accuracy: \033[33m{val_accuracy:.2f}%\033[0m"
                f"\n\033[34m[ \033[36m{current_time} \033[34m- Tempo impiegato: {epoch_time:.2f}s ]\033[0m\n"
            )
            print(last_epoch_output)

            suppress_stdout()
            enable_stdout()

            scheduler.step(val_loss)

            # Passa l'output dell'epoca corrente alla funzione di early stopping
            early_stopping(val_loss, model, last_epoch_output)

            if early_stopping.early_stop:
                print("\033[31mEarly stopping attivato\033[0m")
                break

            if val_accuracy >= target_accuracy:
                print(f"Validation accuracy raggiunta {val_accuracy:.2f}%")
                break
    
    except KeyboardInterrupt:
        print("\033[31mTraining interrotto manualmente\033[0m")
    
    finally:
        minutes, seconds = divmod(sum_time, 60)
        current_time = time.strftime('%H:%M:%S', time.localtime())
        print(f'\n\033[34m[ Tempo totale impiegato: {int(minutes)}m {int(seconds)}s ]\033[0m')
        writer.close()

        # Salva l'output dell'epoca con la migliore accuracy, se esiste
        if early_stopping.best_epoch_output:
            save_epoch_output(early_stopping.best_epoch_output)
