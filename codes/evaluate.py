import torch

def evaluate(model, val_loader, criterion, return_preds=False):
    """
    Valuta il modello sul dataset di validazione.

    Args:
        model (torch.nn.Module): modello da valutare
        val_loader (torch.utils.data.DataLoader): dataloader del dataset di validazione
        criterion (torch.nn.Module): funzione di perdita da utilizzare
        return_preds (bool): se True, restituisce anche le etichette e le predizioni

    Returns:
        val_loss (float): perdita media sul dataset di validazione
        val_accuracy (float): accuratezza del modello sul dataset di validazione
        val_labels (list, opzionale): etichette vere se return_preds è True
        val_preds (list, opzionale): predizioni del modello se return_preds è True
    """
    # Imposta il modello in modalità valutazione
    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    val_labels = []
    val_preds = []
    
    # Disabilita il calcolo del gradiente per migliorare le prestazioni
    with torch.no_grad():
        for inputs, labels in val_loader:
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total_samples += labels.size(0)
            total_correct += (predicted == labels).sum().item()
            
            if return_preds:
                val_labels.extend(labels.cpu().numpy())
                val_preds.extend(predicted.cpu().numpy())
    
    val_loss = total_loss / len(val_loader)
    val_accuracy = 100 * total_correct / total_samples
    
    if return_preds:
        return val_loss, val_accuracy, val_labels, val_preds
    else:
        return val_loss, val_accuracy
