import winsound

# Funzione per emettere un allarme in caso di trigger di early stopping
def alert():
    winsound.Beep(frequency=3000, duration=100)  # Beep di 3000 Hz per 100 ms
    winsound.Beep(frequency=3000, duration=100)  # Beep di 3000 Hz per 100 ms
    winsound.Beep(frequency=5000, duration=100)  # Beep di 5000 Hz per 100 ms