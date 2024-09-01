# Progetto d'Esame "Laboratorio IA"

## Descrizione
Questo progetto implementa un modello di rete neurale per classificazione di immagini
relative al dataset CIFAR-10, il quale contiene circa 60'000 immagini 32x32 a colori suddivise 
appunto in 10 classi.

## Struttura del Progetto
- `config/`: file di configurazione:
  - `config.json`: iper-parametri di configurazione
  - `config_schema.json`: file di strutturazione e tipizzazione dati
    
- `data/`: caricamento e preprocessing dei dati:
  - `data_loader.py`: file di caricamento dei dati che scarica il dataset in caso di prima esecuzione e si occupa di pre-processing
    
- `models/`: definizione del modello:
  - `best_model.pth`: salvataggio del modello migliore in fase di allenamento
    
- `notebooks/`: analisi dei dati.

- `runs/`: archivio dei run di allenamento
  
- `utils/`: funzioni di utilità:
  - `early_stopping.py`: file di gestione di modalità di early_stopping in caso di mancati miglioramenti delle prestazioni di accuracy e loss
  - `alert.py`: utilità sonora in fase di terminazione allenamento
  - `beep.py`: utilità sonorta in fase di miglioramento modello
  - `clear_console.py`: utilità per pulire la console e garantire un'esperienza di esecuzione più "pulita"
  - `console_output_manager.py`: utilità per gestire sopprimere/permettere gli output a terminale per evitare "fastidiosi" messaggi come future-warnings, ...
  - `save_epoch_output.py`: utilità che scrive su un file .txt il risultato migliore dell'ultimo ciclo di allenamento eseguito

- `testing_images/`: archivio di immagini con il quale verificare la corretteza del modello
  
- `main.py`: script principale da cui avvviare l'addestramento del modello tramite comando "python main.py".

- `image_test.py`: script che permettte di verificare/testare il funzionamento del modello sottoponendogli le immagini presenti nella directory "testing_images/"

- `last_output.py`: scritp che permette di leggere il file .txt contenente l'output migliore dell'ultimo addestramento, in modo tale da avere un'idea precisa delle prestazioni

- `environment.yaml`: ambiente contenente tutte le dipendenze necessarie per poter eseguire il codice in ambiente CONDA. Per eseguire il file di installazione eseguire il comando "conda env create -f environment.yaml"
