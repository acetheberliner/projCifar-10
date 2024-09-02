import torch
import torch.optim as optim
import json

# Importa il modello personalizzato e il data loader
from models.model import CustomModel
from data.data_loader import load_data
from utils.clear_console import clear_console
from utils.time_manager import get_current_time

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

    # misura la differenza tra le probabilità previste dal modello e le etichette vere in un problema di classificazione, penalizzando previsioni lontane dalla realtà per migliorare l'accuratezza
    criterion = torch.nn.CrossEntropyLoss(reduction='mean')

    # serve ad aggiornare i pesi del modello durante l'addestramento, riducendo il rischio di overfitting grazie alla regolarizzazione L2 (una tecnica che penalizza i pesi di un modello durante l'addestramento)
    optimizer = optim.AdamW(model.parameters(), lr=config['training']['learning_rate'], weight_decay=1e-4)

    # riduce automaticamente il tasso di apprendimento quando le prestazioni del modello non migliorano per un certo numero di epoche
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)

    # Esegue la funzione di training
    train(model, train_loader, val_loader, criterion, optimizer, scheduler, config)

# Esegue la funzione principale
if __name__ == "__main__":
    clear_console()

    print(f'\033[36mCiclo di allenamento avviato alle ore [ {get_current_time()} ], attendere... \33[0m')
    main()