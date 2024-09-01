
import torch

def evaluate(model, val_loader, criterion):
    """
    Valuta il modello sul dataset di validazione.

    Args:
        model (torch.nn.Module): modello da valutare
        val_loader (torch.utils.data.DataLoader): dataloader del dataset di validazione
        criterion (torch.nn.Module): funzione di perdita da utilizzare

    Returns:
        val_loss (float): perdita media sul dataset di validazione
        val_accuracy (float): accuratezza del modello sul dataset di validazione
    """
    # Imposta il modello in modalità valutazione
    model.eval()

    # Inizializza variabili per tracciare la perdita totale, predizioni corrette e campioni totali
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    
    # Disabilita il calcolo del gradiente per migliorare le prestazioni
    with torch.no_grad():
        # Itera sul dataset di validazione
        for inputs, labels in val_loader:
            # Esegui un passaggio in avanti e ottieni le uscite del modello
            outputs = model(inputs)
            
            # Calcola la perdita tra le uscite e le etichette vere
            loss = criterion(outputs, labels)
            
            # Accumula la perdita totale
            total_loss += loss.item()
            
            # Ottieni le etichette di classe predette
            _, predicted = torch.max(outputs, 1)
            
            # Aggiorna il numero totale di campioni e predizioni corrette
            total_samples += labels.size(0)
            total_correct += (predicted == labels).sum().item()
            
    # Calcola la perdita media e l'accuratezza
    val_loss = total_loss / len(val_loader)
    val_accuracy = 100 * total_correct / total_samples
    
    # Restituisci la perdita media e l'accuratezza
    return val_loss, val_accuracy

