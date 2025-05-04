# Add the parent directory to import sys
import os
import sys
import unittest

import librosa
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from sample_314 import compute_mel_to_audio

filename = librosa.util.example_audio_file()
y, sr = librosa.load(filename)
y = y[:10000]

S = np.abs(librosa.stft(y)) ** 2
M = librosa.feature.melspectrogram(y=y, sr=sr, S=S)
n_fft = 2048
hop_length = 512
win_length = None
window = "hann"
center = True
pad_mode = "reflect"
power = 2.0
n_iter = 32
length = None
dtype = np.float32
np.random.seed(seed=0)
sol = compute_mel_to_audio(
    y,
    sr,
    S,
    M,
    n_fft,
    hop_length,
    win_length,
    window,
    center,
    pad_mode,
    power,
    n_iter,
    length,
    dtype,
)
np.random.seed(seed=0)


def _nnls_obj(x, shape, A, B):
    x = x.reshape(shape)

    diff = np.dot(A, x) - B

    value = 0.5 * np.sum(diff**2)

    grad = np.dot(A.T, diff)

    return value, grad.flatten()


def _nnls_lbfgs_block(A, B, x_init=None, **kwargs):
    if x_init is None:
        x_init = np.linalg.lstsq(A, B, rcond=None)[0]
        np.clip(x_init, 0, None, out=x_init)

    kwargs.setdefault("m", A.shape[1])

    bounds = [(0, None)] * x_init.size
    shape = x_init.shape

    x, obj_value, diagnostics = scipy.optimize.fmin_l_bfgs_b(
        _nnls_obj, x_init, args=(shape, A, B), bounds=bounds, **kwargs
    )
    return x.reshape(shape)


def nnls(A, B, **kwargs):
    if B.ndim == 1:
        return scipy.optimize.nnls(A, B)[0]

    n_columns = int((2**8 * 2**10) // (A.shape[-1] * A.itemsize))

    if B.shape[-1] <= n_columns:
        return _nnls_lbfgs_block(A, B, **kwargs).astype(A.dtype)

    x = np.linalg.lstsq(A, B, rcond=None)[0].astype(A.dtype)
    np.clip(x, 0, None, out=x)
    x_init = x

    for bl_s in range(0, x.shape[-1], n_columns):
        bl_t = min(bl_s + n_columns, B.shape[-1])
        x[:, bl_s:bl_t] = _nnls_lbfgs_block(
            A, B[:, bl_s:bl_t], x_init=x_init[:, bl_s:bl_t], **kwargs
        )
    return x


rng = np.random.seed(seed=0)


def mel_to_stft(M, sr=22050, n_fft=2048, power=2.0, **kwargs):
    mel_basis = librosa.filters.mel(sr, n_fft, n_mels=M.shape[0], **kwargs)
    inverse = nnls(mel_basis, M)
    return np.power(inverse, 1.0 / power, out=inverse)


stft = mel_to_stft(M, sr=sr, n_fft=n_fft, power=power)


def griffinlim(
    S,
    n_iter=32,
    hop_length=None,
    win_length=None,
    window="hann",
    center=True,
    dtype=np.float32,
    length=None,
    pad_mode="reflect",
    momentum=0.99,
    random_state=None,
):
    rng = np.random
    n_fft = 2 * (S.shape[0] - 1)

    angles = np.exp(2j * np.pi * rng.rand(*S.shape))

    rebuilt = 0.0

    for _ in range(n_iter):
        tprev = rebuilt
        inverse = librosa.istft(
            S * angles,
            hop_length=hop_length,
            win_length=win_length,
            window=window,
            center=center,
            dtype=dtype,
            length=length,
        )
        rebuilt = librosa.stft(
            inverse,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            window=window,
            center=center,
            pad_mode=pad_mode,
        )

        angles[:] = rebuilt - (momentum / (1 + momentum)) * tprev
        angles[:] /= np.abs(angles) + 1e-16
    return librosa.istft(
        S * angles,
        hop_length=hop_length,
        win_length=win_length,
        window=window,
        center=center,
        dtype=dtype,
        length=length,
    )


test_sol = griffinlim(
    stft,
    n_iter=n_iter,
    hop_length=hop_length,
    win_length=win_length,
    window=window,
    center=center,
    dtype=dtype,
    length=length,
    pad_mode=pad_mode,
)
assert np.allclose(test_sol, sol)
