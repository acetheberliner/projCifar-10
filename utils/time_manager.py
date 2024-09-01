import time

def get_current_time():
    """Ritorna l'ora corrente in formato HH:MM:SS."""
    return time.strftime('%H:%M:%S', time.localtime())

def calculate_epoch_time(start_time):
    """Calcola il tempo impiegato per completare un'epoca e ritorna il tempo impiegato."""
    return time.time() - start_time

def format_total_time(total_time):
    """Formatta il tempo totale (in secondi) in ore, minuti e secondi."""
    hours, remainder = divmod(total_time, 3600)  # 3600 secondi in un'ora
    minutes, seconds = divmod(remainder, 60)    # 60 secondi in un minuto
    return f'{int(hours)}h {int(minutes)}m {int(seconds)}s'
