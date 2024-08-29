import os
from PIL import Image
import torchvision.transforms as transforms
import torch
import matplotlib.pyplot as plt

from models.model import CustomModel
from utils.clear_console import clear_console

# Funzione per preprocessare l'immagine
def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    image = Image.open(image_path)
    image = transform(image).unsqueeze(0)  # Aggiunge una dimensione batch
    return image

# Funzione per caricare il modello
def load_model(model_path, input_size, output_size):
    model = CustomModel(input_size, output_size)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu'), weights_only=True))
    model.eval()
    return model

# Funzione per effettuare una previsione con il modello
def predict(image, model):
    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output, 1)
    return predicted.item()

# Funzione principale
def main():
    input_size = 32 * 32 * 3  # Dimensione dell'input
    output_size = 10  # Numero di classi

    class_names = ['Aeroplano', 'Automobile', 'Uccello', 'Gatto', 'Cervo/Daino', 'Cane', 'Rana', 'Cavallo', 'Nave/Barca', 'Camion']
    model_path = 'models/best_model.pth'
    base_folder = './testing_images'
    
    model = load_model(model_path, input_size, output_size)
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

            print(f"\n\033[36mSeleziona un'immagine di test della categoria '\033[33m{selected_category}\033[0m \033[36m':\033[0m")
            for i, image in enumerate(images):
                print(f"\033[33m[{i + 1}]\033[0m - {image}")

            img_choice = int(input("\n\033[36mInserisci il \033[33mnumero\033[0m\033[36m associato all'immagine: \033[0m"))
            selected_image = os.path.join(image_folder, images[img_choice - 1])

            image = preprocess_image(selected_image)
            prediction = predict(image, model)
            img = plt.imread(selected_image)

            print(f'\033[32m* Predizione immagine: \033[33m{class_names[prediction]}\033[0m')
            print("\n[Chiudere scheda 'Figure 1' per continuare..]")
            plt.imshow(img)
            plt.show()

        except (ValueError, IndexError):
            print("Scelta non valida. Riprova.")

if __name__ == "__main__":
    main()
