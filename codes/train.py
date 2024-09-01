import torch
import time
from torch.utils.tensorboard import SummaryWriter

from codes.evaluate import evaluate

from utils.early_stopping import EarlyStopping
from utils.console_output_manager import suppress_stdout, enable_stdout
from utils.save_epoch_out import save_epoch_output
from utils.time_manager import get_current_time, calculate_epoch_time, format_total_time

def train(model, train_loader, val_loader, criterion, optimizer, scheduler, config):
    """
    Funzione di training per il modello.
    """
    writer = SummaryWriter('runs')
    epochs = config['training']['epochs']
    early_stopping = EarlyStopping(patience=config['training']['patience'], delta=config['training']['delta'])
    sum_time = 0.0
    target_accuracy = 90.0
    last_epoch_output = ""

    try:
        for epoch in range(epochs):
            start_time = time.time()

            # Inizia addestramento
            model.train()
            running_loss = 0.0
            correct = 0
            total = 0

            # Itera sul dataset di training
            for i, (inputs, labels) in enumerate(train_loader):
                # Calcola la perdita e fa backpropagation
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                # Aggiorna valori di perdita e accuracy
                running_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

            # Calcola la perdita media e l'accuratezza
            train_loss = running_loss / len(train_loader)
            train_accuracy = 100 * correct / total

            # Valuta il modello sul dataset di validazione
            val_loss, val_accuracy = evaluate(model, val_loader, criterion)

            # Scrive i dati su TensorBoard
            writer.add_scalar('Loss/train', train_loss, epoch)
            writer.add_scalar('Accuracy/train', train_accuracy, epoch)
            writer.add_scalar('Loss/validation', val_loss, epoch)
            writer.add_scalar('Accuracy/validation', val_accuracy, epoch)

            # Calcola il tempo impiegato per l'epoca
            epoch_time = calculate_epoch_time(start_time)
            sum_time += epoch_time

            # Stampa l'output dell'epoca
            current_time = get_current_time()

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

            # Gestione early stopping
            suppress_stdout()
            enable_stdout()

            scheduler.step(val_loss)

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
        total_time_formatted = format_total_time(sum_time)
        print(f'\n\033[34m[ Tempo totale impiegato: {total_time_formatted} ]\033[0m')
        writer.close()

        if early_stopping.best_epoch_output:
            save_epoch_output(early_stopping.best_epoch_output)
