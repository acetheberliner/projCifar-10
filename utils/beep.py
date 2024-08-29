import winsound

# Funzione per emettere un beep ad ogni epoca in cui si verifica un miglioramento delle prestazioni di loss
def beep():
    winsound.Beep(frequency=500, duration=100)  # Beep di 500 Hz per 100 ms
    winsound.Beep(frequency=600, duration=100)  # Beep di 600 Hz per 100 ms
