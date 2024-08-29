import torch
import torch.optim as optim
import json

# Importa il modello personalizzato e il data loader
from models.model import CustomModel
from data.data_loader import load_data
from utils.clear_console import clear_console

# Importa la funzione di training
from codes.train import train

# Funzione principale
def main():
    # Carica il file di configurazione
    with open('config/config.json') as f:
        config = json.load(f)

    # Carica i dati di training e validazione
    train_loader, val_loader = load_data(config)

    # Crea un'istanza del modello personalizzato
    model = CustomModel(config['model']['input_size'], config['model']['output_size'])

    # Definisce la funzione di perdita (CrossEntropyLoss)
    criterion = torch.nn.CrossEntropyLoss(reduction='mean')  # Imposta la riduzione della perdita a somma (non media)

    # spiegazione: la funzione CrossEntropyLoss è una funzione di perdita (o loss) che è utilizzata per calcolare la perdita in un problema di classificazione multiclasse.
    # In questo caso, il criterio di calcolo della perdita è definito come la somma (non la media) del logaritmo del valore di verità per ogni classe.
    # In altre parole, la perdita è la somma del logaritmo del valore di verità per ogni esempio.
    # Questo può aumentare la precisione del modello, in quanto riduce la sensibilità rispetto alla perdita media.
    # Nota: l'utilizzo della somma invece della media è utile nel caso in cui vogliamo utilizzare lo scheduler di learning rate per ridurre la velocità di apprendimento.
    # In questo caso, utilizzando la somma si ottiene una misura più accurata della perdita totale.
    # Nota2: è possibile modificare la funzione di perdita per ottenere una misura della perdita media, basta rimuovere il parametro 'reduction' oppure impostarlo a 'elementwise_mean'.


    # Crea un'istanza dell'ottimizzatore AdamW con regolarizzazione L2 (weight decay)
    # L'aggiunta di una leggera regolarizzazione può aiutare a prevenire l'overfitting e migliorare la generalizzazione
    # optimizer = optim.AdamW(model.parameters(), lr=config['training']['learning_rate'], weight_decay=1e-4)
    optimizer = optim.AdamW(model.parameters(), lr=config['training']['learning_rate'], weight_decay=5e-4) #ex 1e-4


    # Crea un'istanza dello scheduler di learning rate (CosineAnnealingLR) che riduce
    # la velocità di apprendimento ogni 16 epoche. Dopo 16 epoche, riduce ulteriormente
    # la velocità di apprendimento se la loss non migliora per 2 epoche
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=16)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

    # Esegue la funzione di training
    train(model, train_loader, val_loader, criterion, optimizer, scheduler, config)

# Esegue la funzione principale se il file è eseguito direttamente
if __name__ == "__main__":
    clear_console()

    print(f'\033[36mCiclo di allenamento avviato, attendere... \33[0m')
    main()