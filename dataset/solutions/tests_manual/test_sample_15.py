# Add the parent directory to import sys
import os
import sys
import unittest

import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from sample_15 import stft

audio_signal = torch.rand(1024)
n_fft = 128
expected_shape = (65, 33, 2)
assert stft(audio_signal, n_fft).shape == expected_shape
