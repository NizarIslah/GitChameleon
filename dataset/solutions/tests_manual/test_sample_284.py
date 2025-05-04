# Add the parent directory to import sys
import os
import sys
import tempfile
import unittest

import librosa
import numpy as np
import soundfile as sf

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sample_284 import compute_stream


filename = librosa.util.example_audio_file()
y, sr = librosa.load(filename)

n_fft = 4096
hop_length = n_fft // 2
stream, stream_blocks = compute_stream(y, sr, n_fft, hop_length)
sol_stream = librosa.stream(
    filename,
    block_length=16,
    frame_length=n_fft,
    hop_length=hop_length,
    mono=True,
    fill_value=0,
)
for c, y_block in enumerate(sol_stream):
    assert np.array_equal(
        librosa.stft(y_block, n_fft=n_fft, hop_length=hop_length, center=False),
        stream_blocks[c],
    )
