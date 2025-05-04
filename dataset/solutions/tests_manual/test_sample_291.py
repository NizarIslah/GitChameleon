import os

# Add the parent directory to the path so we can import the module
import sys
import unittest

import librosa
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sample_291 import compute_plp

filename = librosa.util.example_audio_file()
y, sr = librosa.load(filename)
hop_length = 512
win_length = 384
tempo_min = None
tempo_max = None
onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)


sol = compute_plp(y, sr, hop_length, win_length, tempo_min, tempo_max, onset_env)

ftgram = stft(onset_env, n_fft=win_length, hop_length=1, center=True, window="hann")

tempo_frequencies = np.fft.rfftfreq(n=win_length, d=(sr * 60 / float(hop_length)))

ftmag = np.abs(ftgram)
peak_values = ftmag.max(axis=0, keepdims=True)
ftgram[ftmag < peak_values] = 0

ftgram[:] /= peak_values

pulse = istft(ftgram, hop_length=1, length=len(onset_env))

np.clip(pulse, 0, None, pulse)
test_sol = librosa.util.normalize(pulse)
assert np.array_equal(test_sol, sol)
