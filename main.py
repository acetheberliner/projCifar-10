import torch
import torch.optim as optim
import json
import os

# Importa il modello personalizzato e il data loader
from models.model import CustomModel
from data.data_loader import load_data

from utils.clear_console import clear_console
from utils.time_manager import get_current_time
from codes.train import load_checkpoint

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
    model = CustomModel(config['model']['output_size'])

    # misura la differenza tra le probabilità previste dal modello e le etichette vere in un problema di classificazione
    criterion = torch.nn.CrossEntropyLoss(reduction='mean')

    # serve ad aggiornare i pesi del modello durante l'addestramento
    optimizer = optim.AdamW(model.parameters(), lr=config['training']['learning_rate'], weight_decay=1e-4)
    # optimizer = optim.Adagrad(model.parameters(), lr=config['training']['learning_rate'], weight_decay=5e-4) <----------- AdaGrad

    # riduce automaticamente il tasso di apprendimento quando le prestazioni del modello non migliorano per un certo numero di epoche
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=2)
    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5) <----------- per AdaGrad (impostare inoltre in config.json lr=0.0025 e batch_size=256)

    # Chiede all'utente se vuole iniziare da un checkpoint o da zero
    choice = input("\nAvviare un nuovo addestramento [\033[35mN\033[0m] | Riprendere dall'ultimo checkpoint [\033[35mR\033[0m]: ").strip().lower()

    if choice == "r" and os.path.isfile('models/best_model.pth'):
        print("\033[33mCaricamento checkpoint...\033[0m\n")
        start_epoch = load_checkpoint(model, optimizer, 'models/best_model.pth')
        print(f"Ripresa addestramento da epoca {start_epoch + 1}")
    else:
        print("\033[33mAvvio di un nuovo addestramento...\033[0m\n")
        start_epoch = 0

    # Esegue la funzione di training
    train(model, train_loader, val_loader, criterion, optimizer, scheduler, config, start_epoch)

# Esegue la funzione principale
if __name__ == "__main__":
    clear_console()

    print(f'\033[36mCiclo di allenamento avviato alle ore [\033[0m \033[33m{get_current_time()}\033[0m \033[36m]\033[0m')
    main()
