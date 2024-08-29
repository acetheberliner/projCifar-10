# ------------------------------------------------------------------------------------------------------------------------------------------------
#                CON QUESTO CODICE DI MODELLO SI RAGGIUNGE CIRCA 85% DI VALIATION ACCURACY... IN CIRCA UN'ORA DI TRAINING
#                                    (considerando che ogni epoch impiega circa 3 minuti per completarsi)
# ------------------------------------------------------------------------------------------------------------------------------------------------
# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# class CustomModel(nn.Module):
#     def __init__(self, input_size, output_size):
#         super(CustomModel, self).__init__()
#         self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)  # Aumenta i filtri a 64
#         self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)  # Aumenta i filtri a 128
#         self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
#         self.conv4 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)  # Aumenta i filtri a 512
#         self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

#         # Dato che l'output del quarto layer è 512 feature map di 2x2, l'input a fc1 sarà 512 * 2 * 2 = 2048
#         self.fc1 = nn.Linear(512 * 2 * 2, 1024)  # Aumenta la dimensione del primo fully connected
#         self.fc2 = nn.Linear(1024, 512)  # Aggiungi un altro fully connected
#         self.fc3 = nn.Linear(512, output_size)

#         self.dropout = nn.Dropout(p=0.5)
        
#     def forward(self, x):
#         self.bn1 = nn.BatchNorm2d(64)
#         self.bn2 = nn.BatchNorm2d(128)
#         self.bn3 = nn.BatchNorm2d(256)
#         self.bn4 = nn.BatchNorm2d(512)

#         # Nel forward pass, inserisci batch normalization dopo ogni convoluzione e prima dell'attivazione ReLU
#         x = self.pool(F.relu(self.bn1(self.conv1(x))))
#         x = self.pool(F.relu(self.bn2(self.conv2(x))))
#         x = self.pool(F.relu(self.bn3(self.conv3(x))))
#         x = self.pool(F.relu(self.bn4(self.conv4(x))))

#         x = x.view(x.size(0), -1)  # Flatten

#         x = F.relu(self.fc1(x))
#         x = self.dropout(x)
#         x = F.relu(self.fc2(x))
#         x = self.fc3(x)
#         return x

# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torchvision import models

# ------------------------------------------------------------------------------------------------------------------------------------------------
#               CON IL MODELLO EFFICIENTNET PRE ADDESTRATO SI GARANTISCE UNA MAGGIORE VELOCITA' DI TRAINING
# ------------------------------------------------------------------------------------------------------------------------------------------------

# Importa il modello EfficientNet-B0 e i pesi pre-addestrati da torchvision
from torchvision.models import efficientnet_b2, EfficientNet_B2_Weights
import torch.nn as nn

# Definisce una classe per il modello personalizzato
class CustomModel(nn.Module):
    # Inizializza il modello con la dimensione dell'input e dell'output
    def __init__(self, input_size, output_size):
        # Chiama il costruttore della classe base
        super(CustomModel, self).__init__()
        
        # Carica il modello EfficientNet-B0 con i pesi pre-addestrati su ImageNet
        self.base_model = efficientnet_b2(weights=EfficientNet_B2_Weights.IMAGENET1K_V1)
        
        # Blocca i parametri dei primi strati convolutivi per il transfer learning
        # for param in self.base_model.features.parameters():
        #     param.requires_grad = False

        # Modifica l'ultimo livello del classificatore per adattarlo alla dimensione dell'output desiderata
        # self.base_model.classifier[1] = nn.Linear(self.base_model.classifier[1].in_features, output_size)

        # Modifica l'ultimo livello del classificatore per adattarlo alla dimensione dell'output desiderata
        self.base_model.classifier[1] = nn.Sequential(
            nn.Dropout(0.7),
            nn.Linear(self.base_model.classifier[1].in_features, 512),  # Aggiungi un fully connected layer
            nn.ReLU(),  # Aggiungi una funzione di attivazione ReLU
            nn.Dropout(0.6),  # Aggiungi un altro livello di dropout ex 0.5
            nn.Linear(512, output_size)  # Output finale
        )

    # Definisce il comportamento del modello durante la propagazione in avanti
    def forward(self, x):
        # Passa l'input attraverso il modello base
        x = self.base_model(x)
        
        return x

