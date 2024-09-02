# Progetto d'Esame "Laboratorio IA"
Bagnolini Tommaso

## Descrizione
In questo progetto, ho affrontato il problema della classificazione delle immagini utilizzando il dataset CIFAR-10, che contiene 60.000 immagini 32x32 suddivise in 10 classi distinte.
(https://www.cs.toronto.edu/~kriz/cifar.html)

![Immagine1](https://github.com/user-attachments/assets/27d18f0c-ed73-4a38-b7ee-2528aac291d4)

## Struttura del Progetto
- `config/`: file di configurazione:
  - `config.json`: iper-parametri di configurazione
  - `config_schema.json`: file di strutturazione e tipizzazione dati
    
- `data/`: caricamento e preprocessing dei dati:
  - `data_loader.py`: file di caricamento dei dati che scarica il dataset in caso di prima esecuzione e si occupa di pre-processing
    
- `models/`: definizione del modello:
  - `model.py`: file contenente la struttura del modello basato su EfficientNet
  - `best_model.pth`: salvataggio del modello migliore in fase di allenamento
    
- `notebooks/`: notebook .ipynb di analisi preventiva dei dati del dataset.

- `runs/`: archivio dei run di allenamento
  
- `utils/`: funzioni di utilità:
  - `early_stopping.py`: file di gestione di modalità di early_stopping in caso di mancati miglioramenti delle prestazioni di accuracy e loss per un predeterminato periodo massimo di epoche chiamato “patience”.
  - `alert.py`: utilità sonora in fase di terminazione allenamento visualizzabili tramite tensorboard
  - `beep.py`: utilità sonora in fase di miglioramento modello
  - `clear_console.py`: utilità per ripulire la console e garantire un'esperienza di esecuzione più completa e fluida.
  - `console_output_manager.py`: utilità per sopprimere/consentire gli output a terminale per evitare "fastidiosi" messaggi come future-warnings, ecc...
  - `save_epoch_output.py`: utilità che scrive su un file .txt il risultato migliore dell'ultimo ciclo di allenamento eseguito
  - `time_manager.py`: utilità per la gestione del tempo all'interno del programma

- `testing_images/`: archivio di immagini con il quale verificare la correttezza del modello
  
- `main.py`: script principale da cui avviare l'addestramento del modello tramite comando "python main.py".

- `image_test.py`: script che permettte di verificare/testare il funzionamento del modello sottoponendogli le immagini presenti nella directory "testing_images/", utilizzabile tramite comando “python image_test.py”

- `last_output.py`: script che permette di leggere il file .txt contenente l'output migliore dell'ultimo addestramento, in modo tale da avere un'idea precisa delle prestazioni

- `environment.yaml`: ambiente contenente tutte le dipendenze necessarie per poter eseguire il codice in ambiente CONDA. Per eseguire il file di installazione eseguire il comando "conda env create -f environment.yaml" (posizionandosi prima nella root del repository)
