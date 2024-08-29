import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

from utils.console_output_manager import suppress_stdout
from utils.console_output_manager import enable_stdout

# Funzione per caricare i dati
def load_data(config):
    suppress_stdout()

    # Definisce le trasformazioni per il training set
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
        transforms.ToTensor(),  # Converte l'immagine in un tensore di PyTorch
        # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


    # Definisce le trasformazioni per il validation set
    val_transform = transforms.Compose([
        transforms.Resize(224),  # Resize per EfficientNet
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Carica il dataset CIFAR-10
    train_dataset = datasets.CIFAR10(root='data', train=True, download=True, transform=train_transform)
    val_dataset = datasets.CIFAR10(root='data', train=False, download=True, transform=val_transform)

    # Divisione del train set in training e validation set, se necessario
    if config['data']['validation_split'] > 0:
        val_size = int(config['data']['validation_split'] * len(train_dataset))
        train_size = len(train_dataset) - val_size
        train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])

    # Crea i DataLoader per il training e validation set
    num_workers = 8 if torch.cuda.is_available() else 0
    train_loader = DataLoader(train_dataset, batch_size=config['data']['batch_size'], shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=config['data']['batch_size'], shuffle=False)

    enable_stdout()

    return train_loader, val_loader
