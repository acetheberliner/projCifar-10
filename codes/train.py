import torch
import time
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

from codes.evaluate import evaluate
from utils.early_stopping import EarlyStopping
from utils.console_output_manager import suppress_stdout, enable_stdout
from utils.save_epoch_out import save_epoch_output
from utils.time_manager import get_current_time, calculate_epoch_time, format_total_time
from utils.class_names import class_names

# Definizione della funzione load_checkpoint
def load_checkpoint(model, optimizer, filepath):
    checkpoint = torch.load(filepath, weights_only=True)
    
    model_state_dict = model.state_dict()
    checkpoint_state_dict = checkpoint['model_state_dict']
    
    filtered_state_dict = {k: v for k, v in checkpoint_state_dict.items() if k in model_state_dict}
    
    model_state_dict.update(filtered_state_dict)
    model.load_state_dict(model_state_dict)
    
    if 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    start_epoch = checkpoint.get('epoch', 0)
    return start_epoch

# Funzione per calcolare e salvare la Confusion Matrix
def save_confusion_matrix(labels, preds, class_names, epoch, phase, writer):
    # Converti labels e preds in array numpy, se non lo sono già
    labels = np.array(labels)
    preds = np.array(preds)
    
    # Assicurati che class_names sia una sequenza
    if not isinstance(class_names, (list, np.ndarray)):
        raise ValueError("class_names deve essere una sequenza (lista o array numpy).")

    # Calcola la Confusion Matrix
    cm = confusion_matrix(labels, preds)
    
    # Crea il ConfusionMatrixDisplay
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    
    # Traccia la Confusion Matrix con il cmap di colore
    disp.plot(cmap=plt.cm.Blues)
    plt.title(f'Confusion Matrix - {phase} - Epoch {epoch + 1}')
    
    # Salva la figura come immagine in TensorBoard
    writer.add_figure(f'Confusion Matrix/{phase}', disp.figure_, epoch)
    plt.close('all')

# Definizione della funzione train
def train(model, train_loader, val_loader, criterion, optimizer, scheduler, config, start_epoch=0):
    writer = SummaryWriter('runs')
    epochs = config['training']['epochs']
    early_stopping = EarlyStopping(patience=config['training']['patience'], delta=config['training']['delta'])
    sum_time = 0.0
    last_epoch_output = ""

    inputs, _ = next(iter(train_loader))
    writer.add_graph(model, inputs)

    try:
        for epoch in range(start_epoch, epochs):
            start_time = time.time()

            model.train()
            running_loss = 0.0
            correct = 0
            total = 0
            all_labels = []
            all_preds = []

            print("-------------------------------------------------------------------------------------------------------")

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

                # Salva le etichette e le predizioni per la confusion matrix
                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(predicted.cpu().numpy())

            train_loss = running_loss / len(train_loader)
            train_accuracy = 100 * correct / total

            save_confusion_matrix(all_labels, all_preds, class_names, epoch, 'Training', writer) # Salva la confusion matrix per il training
            val_loss, val_accuracy, val_labels, val_preds = evaluate(model, val_loader, criterion, return_preds=True) # Validazione
            save_confusion_matrix(val_labels, val_preds, class_names, epoch, 'Validation', writer) # Salva la confusion matrix per la validazione 

            # Log dei risultati su TensorBoard
            writer.add_scalar('Loss/train', train_loss, epoch)
            writer.add_scalar('Accuracy/train', train_accuracy, epoch)
            writer.add_scalar('Loss/validation', val_loss, epoch)
            writer.add_scalar('Accuracy/validation', val_accuracy, epoch)
            writer.add_scalar('Learning Rate', scheduler.get_last_lr()[0], epoch)

            # Log delle distribuzioni dei pesi e dei gradienti
            for name, param in model.named_parameters():
                writer.add_histogram(f'{name}.weight', param, epoch)
                if param.grad is not None:
                    writer.add_histogram(f'{name}.grad', param.grad, epoch)

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

            early_stopping(val_loss, model, optimizer, epoch)

            if early_stopping.early_stop:
                print("\033[31mEarly stopping attivato\033[0m")
                break
    
    except KeyboardInterrupt:
        print("\n\033[31mTraining interrotto manualmente\033[0m")
        if last_epoch_output:
            save_epoch_output(last_epoch_output)  # Salva l'output dell'ultima epoca prima dell'interruzione
            raise

    finally:
        total_time_formatted = format_total_time(sum_time)
        print(f'\n\033[34m[ Tempo totale impiegato: {total_time_formatted} ]\033[0m')
        if not writer._closed:
            try:
                writer.close()
            except Exception as e:
                print(f"Errore durante la chiusura di TensorBoard: {e}")
        if early_stopping.best_epoch_output:
            save_epoch_output(early_stopping.best_epoch_output)
