# Add the parent directory to import sys
import os
import sys
import unittest

import librosa
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from sample_310 import compute_vqt


filename = librosa.util.example_audio_file()
y, sr = librosa.load(filename)

sol = compute_vqt(y, sr)
test_sol = librosa.vqt(y, sr=sr)
assert np.allclose(test_sol, sol)
