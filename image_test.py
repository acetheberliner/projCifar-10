import os
import torch
import matplotlib.pyplot as plt
import torchvision.transforms as transforms

from PIL import Image  # Libreria per leggere e manipolare le immagini
from models.model import CustomModel
from utils.clear_console import clear_console

# Funzione per preprocessare l'immagine
def preprocess_image(image_path):
    """
    Preprocessa l'immagine per renderla adatta all'input del modello.
    """
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    image = Image.open(image_path)
    image = transform(image).unsqueeze(0)  # Aggiunge una dimensione batch
    return image

# Funzione per caricare il modello dal checkpoint
def load_model(model_path):
    """
    Carica il modello e lo stato dell'ottimizzatore dal file di checkpoint specificato.
    """
    model = CustomModel(output_size=10)  # Specifica il numero di classi per il modello CIFAR-10
    checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()  # Imposta il modello in modalità valutazione
    return model

# Funzione per effettuare una previsione con il modello
def predict(image, model):
    """
    Effettua una previsione sull'immagine specificata utilizzando il modello.
    Restituisce la classe predetta.
    """
    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output, 1)
    return predicted.item()

# Funzione principale
def main():
    """
    Funzione principale che gestisce l'input utente e effettua le previsioni.
    """
    class_names = ['Aeroplano', 'Automobile', 'Uccello', 'Gatto', 'Cervo/Daino', 'Cane', 'Rana', 'Cavallo', 'Nave/Barca', 'Camion']
    model_path = 'models/best_model.pth'
    base_folder = './testing_images'
    
    model = load_model(model_path)
    categories = [d for d in os.listdir(base_folder) if os.path.isdir(os.path.join(base_folder, d))]

    while True:
        clear_console()

        print(f"\033[36m------------------ MENU ------------------\033[0m")
        print(f"\033[36mSeleziona una \033[32mcategoria\033[0m \033[36mdi CIFAR-10:\033[0m")
        for i, category in enumerate(categories):
            print(f"\033[33m[{i + 1}]\033[0m - {category.capitalize()}")
        print(f"\033[36m------------------------------------------\033[0m")

        print(f"\033[31m[0] - Chiudi\033[0m")

        try:
            choice = int(input("\n\033[36mInserisci il numero associato alla categoria: \033[0m"))
            if choice == 0:
                print(f"\033[31mUscita dal programma\033[0m")
                break

            selected_category = categories[choice - 1]
            image_folder = os.path.join(base_folder, selected_category)
            images = [f for f in os.listdir(image_folder) if os.path.isfile(os.path.join(image_folder, f))]

            print(f"\033[36mSeleziona un'immagine di test della categoria '\033[33m{selected_category}\033[0m \033[36m':\033[0m")
            for i, image in enumerate(images):
                print(f"\033[33m[{i + 1}]\033[0m - {image}")

            img_choice = int(input("\n\033[36mInserisci il \033[33mnumero\033[0m\033[36m associato all'immagine: \033[0m"))
            selected_image = os.path.join(image_folder, images[img_choice - 1])

            image = preprocess_image(selected_image)
            prediction = predict(image, model)
            img = plt.imread(selected_image)

            print(f'\033[32m* Predizione immagine: \033[33m{class_names[prediction]}\033[0m')
            print("\n[Chiudere scheda 'Immagine selezionata' per continuare]")
            plt.imshow(img)
            
            # Ottieni la figura corrente e imposta il titolo della finestra
            plt.gcf().canvas.manager.set_window_title("Immagine selezionata")
            plt.gcf().set_facecolor("white")
            plt.axis("off")
            plt.grid(False)
            plt.title(f"{images[img_choice - 1]}")            
            plt.show()

        except (ValueError, IndexError):
            print("Scelta non valida. Riprova.")

if __name__ == "__main__":
    main()
