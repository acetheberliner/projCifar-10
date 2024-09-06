import torch.nn as nn
from torchvision.models import efficientnet_b2, efficientnet_b0, EfficientNet_B2_Weights, EfficientNet_B0_Weights

from torchvision.models import mobilenet_v2, MobileNet_V2_Weights

class CustomModel(nn.Module):
    def __init__(self, output_size):
        super(CustomModel, self).__init__()
        # Carica il modello EfficientNet-B2 pre-addestrato
        self.base_model = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
        
        # Modifica l'ultimo layer per il numero di output desiderato
        self.base_model.classifier[1] = nn.Linear(self.base_model.classifier[1].in_features, output_size)

    def forward(self, x):
        return self.base_model(x)
    
    # def __init__(self, output_size):
    #     super(CustomModel, self).__init__()
    #     self.base_model = mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT)
    #     self.base_model.classifier[1] = nn.Linear(self.base_model.classifier[1].in_features, output_size)

    # def forward(self, x):
    #     return self.base_model(x)


# class CustomModel(nn.Module):
#     """
#     Modello personalizzato che utilizza EfficientNet-B2 come base
#     e modifica l'ultima layer per adattarla al numero di output desiderato.
#     """
#     def __init__(self, input_size, output_size):
#         # Inizializza il modello personalizzato.
#         super(CustomModel, self).__init__()
        
#         # Carica il modello base EfficientNet-B2
#         self.base_model = efficientnet_b2(weights=EfficientNet_B2_Weights.IMAGENET1K_V1)

#         self.base_model.classifier[1] = nn.Sequential(
#             nn.Dropout(0.5),
#             nn.Linear(self.base_model.classifier[1].in_features, 512),
#             nn.BatchNorm1d(512),
#             nn.ReLU(),
#             nn.Dropout(0.4),
#             nn.Linear(512, 256),
#             nn.BatchNorm1d(256),
#             nn.ReLU(),
#             nn.Dropout(0.3),
#             nn.Linear(256, output_size)
#         )

#     def forward(self, x):
#         # Passa l'input attraverso il modello base
#         x = self.base_model(x)
#         return x

