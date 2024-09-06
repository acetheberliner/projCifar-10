import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

from utils.console_output_manager import suppress_stdout
from utils.console_output_manager import enable_stdout

# Funzione per caricare i dati
def load_data(config):
    """
    Carica il dataset CIFAR-10 e crea i DataLoader per il training e validation set.

    Args:
    config (dict): dizionario contenente le informazioni di configurazione per l'addestramento del modello.

    Returns:
    train_loader (DataLoader): oggetto per iterare sul dataset di training.
    val_loader (DataLoader): oggetto per iterare sul dataset di validazione.
    """
    suppress_stdout()

    # Definisce le trasformazioni per le immagini
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(), # Ruota l'immagine di 90 gradi in senso orario
        transforms.RandomCrop(32, padding=4), # Esegue un crop di 32x32
        transforms.RandomRotation(15), # Ruota l'immagine di 15 gradi in senso orario
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1), # Modifica i parametri delle immagini
        transforms.ToTensor(), # Converte l'immagine in un tensore
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # Normalizza i valori dei tensori
    ])

    val_transform = transforms.Compose([
        transforms.Resize(224), # Rimappa le immagini a 224x224
        transforms.ToTensor(), # Converte l'immagine in un tensore
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), # Normalizza i valori dei tensori
    ])

    # Carica il dataset CIFAR-10
    train_dataset = datasets.CIFAR10(root='data', train=True, download=True, transform=train_transform)
    val_dataset = datasets.CIFAR10(root='data', train=False, download=True, transform=val_transform)

    # Divisione del train set in training e validation set, se necessario
    if config['data']['validation_split'] > 0:
        val_size = int(config['data']['validation_split'] * len(train_dataset))
        train_size = len(train_dataset) - val_size
        train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42))

    # Crea i DataLoader per il training e validation set
    train_loader = DataLoader(train_dataset, batch_size=config['data']['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config['data']['batch_size'], shuffle=False)

    enable_stdout()

    return train_loader, val_loader

