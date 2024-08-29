import torch

def evaluate(model, val_loader, criterion):
    # Setta il modello in modalità valutazione
    model.eval()

    # Inizializza variabili per tracciare la perdita totale, predizioni corrette e campioni totali
    val_loss = 0.0
    correct = 0
    total = 0
    
    # Disabilita il calcolo del gradiente per migliorare le prestazioni
    with torch.no_grad():
        # Itera sul dataset di validazione
        for inputs, labels in val_loader:
            # Passaggio in avanti: ottieni le uscite del modello
            outputs = model(inputs)
            
            # Calcola la perdita tra le uscite e le etichette vere
            loss = criterion(outputs, labels)
            
            # Accumula la perdita totale
            val_loss += loss.item()
            
            # Ottieni le etichette di classe predette
            _, predicted = torch.max(outputs, 1)
            
            # Aggiorna il numero totale di campioni e predizioni corrette
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
    # Calcola la perdita media e l'accuratezza
    val_loss = val_loss / len(val_loader)
    val_accuracy = 100 * correct / total
    
    # Restituisci la perdita media e l'accuratezza
    return val_loss, val_accuracy
