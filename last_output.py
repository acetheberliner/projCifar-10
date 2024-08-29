def view_last_epoch_output(filename="epoch_output.txt"):
    """Legge e visualizza l'output dell'ultima epoca salvato in un file .txt."""
    try:
        with open(filename, "r") as f:
            content = f.read()
            print(content)
    except FileNotFoundError:
        print(f"Il file {filename} non è stato trovato. Assicurati di aver completato almeno un ciclo di training.")

if __name__ == "__main__":
    view_last_epoch_output()
