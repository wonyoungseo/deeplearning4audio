import numpy as np
import librosa, librosa.display
import matplotlib.pyplot as plt

FIG_SIZE = (15, 10)

file = "source/blues.00000.wav"

signal, sample_rate = librosa.load(file, sr=22050)