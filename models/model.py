import torch.nn as nn
from torchvision.models import efficientnet_b2, EfficientNet_B2_Weights

class CustomModel(nn.Module):
    """
    Modello personalizzato che utilizza EfficientNet-B2 come base
    e modifica l'ultima layer per adattarla al numero di output desiderato.
    """
    def __init__(self, input_size, output_size):
        # Inizializza il modello personalizzato.
        super(CustomModel, self).__init__()
        
        # Carica il modello base EfficientNet-B2
        self.base_model = efficientnet_b2(weights=EfficientNet_B2_Weights.IMAGENET1K_V1)
        
        # Modifica l'ultima layer per adattarla al numero di output desiderato
        self.base_model.classifier[1] = nn.Sequential(
            nn.Dropout(0.7), # Dropout per regolarizzare
            nn.Linear(self.base_model.classifier[1].in_features, 1024), # Linear layer con 1024 neuroni
            nn.BatchNorm1d(1024), # Batch normalization
            nn.ReLU(), # Funzione di attivazione ReLU
            nn.Dropout(0.6), # Dropout per regolarizzare
            nn.Linear(1024, 512), # Linear layer con 512 neuroni
            nn.BatchNorm1d(512), # Batch normalization
            nn.ReLU(), # Funzione di attivazione ReLU
            nn.Dropout(0.5), # Dropout per regolarizzare
            nn.Linear(512, output_size) # Linear layer con output_size neuroni
        )

    def forward(self, x):
        # Passa l'input attraverso il modello base
        x = self.base_model(x)
        return x

