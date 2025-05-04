import unittest
import os
import numpy as np
import tempfile
import soundfile as sf
import librosa
import sys
import io
from unittest.mock import patch, MagicMock

# Add the parent directory to the path so we can import the module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import sample_283


filename = librosa.util.example_audio_file()
y, sr = librosa.load(filename)

n_fft = 4096
hop_length = n_fft // 2

stream, stream_blocks = compute_stream(y, sr, n_fft, hop_length)
sol_stream = sf.blocks(
    filename,
    blocksize=n_fft + 15 * hop_length,
    overlap=n_fft - hop_length,
    fill_value=0,
)
sol_blocks = []
for c, block in enumerate(sol_stream):
    y = librosa.to_mono(block.T)
    D = librosa.stft(y, n_fft=n_fft, hop_length=hop_length, center=False)
    sol_blocks.append(D)
for i in range(0, len(stream_blocks)):
    assert np.array_equal(sol_blocks[i], stream_blocks[i])
