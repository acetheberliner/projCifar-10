import sys
import os

def suppress_stdout():
    sys.stdout = open(os.devnull, 'w')

def enable_stdout():
    sys.stdout = sys.__stdout__