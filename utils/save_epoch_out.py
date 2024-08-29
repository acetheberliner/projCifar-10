def save_epoch_output(output, filename="epoch_output.txt"):
    """Salva l'output dell'ultima epoca in un file .txt."""
    with open(filename, "w") as f:
        f.write(output)