#!/usr/bin/env python
# Test file for sample_311.py

import os

# Add the parent directory to the path so we can import the sample
import sys
import unittest

import librosa
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from sample_311 import compute_griffinlim_cqt


filename = librosa.util.example_audio_file()
y, sr = librosa.load(filename)
y = y[:10000]

C = np.abs(librosa.cqt(y=y, sr=sr, bins_per_octave=36, n_bins=7 * 36))
n_iter = 32
hop_length = 512
fmin = None
bins_per_octave = 36
tuning = 0.0
filter_scale = 1
norm = 1
sparsity = 0.01
window = "hann"
scale = True
pad_mode = "reflect"
res_type = "kaiser_fast"
dtype = None
length = None
momentum = 0.99
init = None
rng = np.random.RandomState(seed=0)
sol = compute_griffinlim_cqt(
    y,
    sr,
    C,
    n_iter,
    hop_length,
    fmin,
    bins_per_octave,
    tuning,
    filter_scale,
    norm,
    sparsity,
    window,
    scale,
    pad_mode,
    res_type,
    dtype,
    length,
    momentum,
    init,
)

if fmin is None:
    fmin = librosa.note_to_hz("C1")

angles = np.empty(C.shape, dtype=np.complex64)
if init == "random":
    angles[:] = np.exp(2j * np.pi * rng.rand(*C.shape))
elif init is None:
    angles[:] = 1.0
rebuilt = 0.0

for _ in range(n_iter):
    tprev = rebuilt

    inverse = librosa.constantq.icqt(
        C * angles,
        sr=sr,
        hop_length=hop_length,
        bins_per_octave=bins_per_octave,
        fmin=fmin,
        tuning=tuning,
        filter_scale=filter_scale,
        window=window,
        length=length,
        res_type=res_type,
    )

    rebuilt = librosa.constantq.cqt(
        inverse,
        sr=sr,
        bins_per_octave=bins_per_octave,
        n_bins=C.shape[0],
        hop_length=hop_length,
        fmin=fmin,
        tuning=tuning,
        filter_scale=filter_scale,
        window=window,
        res_type=res_type,
    )
    angles[:] = rebuilt - (momentum / (1 + momentum)) * tprev
    angles[:] /= np.abs(angles) + 1e-16

test_sol = librosa.constantq.icqt(
    C * angles,
    sr=sr,
    hop_length=hop_length,
    bins_per_octave=bins_per_octave,
    tuning=tuning,
    filter_scale=filter_scale,
    fmin=fmin,
    window=window,
    length=length,
    res_type=res_type,
)
assert np.allclose(test_sol, sol)
