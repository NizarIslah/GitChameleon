# Test file for sample_309.py
import os

# Add the parent directory to the path so we can import the sample
import sys
import unittest

import librosa
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sample_309 import compute_vqt


filename = librosa.util.example_audio_file()
y, sr = librosa.load(filename)
hop_length = 512
fmin = None
n_bins = 84
gamma = None
bins_per_octave = 12
tuning = 0.0
filter_scale = 1
norm = 1
sparsity = 0.01
window = "hann"
scale = True
pad_mode = "reflect"
res_type = None
dtype = None

sol = compute_vqt(
    y,
    sr,
    hop_length,
    fmin,
    n_bins,
    gamma,
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
)


def dtype_r2c(d, default=np.complex64):
    mapping = {
        np.dtype(np.float32): np.complex64,
        np.dtype(np.float64): np.complex128,
        np.dtype(np.float): np.complex,
    }

    dt = np.dtype(d)
    if dt.kind == "c":
        return dt

    return np.dtype(mapping.get(dt, default))


n_octaves = int(np.ceil(float(n_bins) / bins_per_octave))
n_filters = min(bins_per_octave, n_bins)

len_orig = len(y)

alpha = 2.0 ** (1.0 / bins_per_octave) - 1

if fmin is None:
    fmin = librosa.note_to_hz("C1")

if tuning is None:
    tuning = librosa.pitch.estimate_tuning(y=y, sr=sr, bins_per_octave=bins_per_octave)

if gamma is None:
    gamma = 24.7 * alpha / 0.108

if dtype is None:
    dtype = dtype_r2c(y.dtype)

fmin = fmin * 2.0 ** (tuning / bins_per_octave)

freqs = librosa.time_frequency.cqt_frequencies(
    n_bins, fmin, bins_per_octave=bins_per_octave
)[-bins_per_octave:]

fmin_t = np.min(freqs)
fmax_t = np.max(freqs)

Q = float(filter_scale) / alpha
filter_cutoff = (
    fmax_t * (1 + 0.5 * librosa.filters.window_bandwidth(window) / Q) + 0.5 * gamma
)
nyquist = sr / 2.0

auto_resample = False
if not res_type:
    auto_resample = True
    if filter_cutoff < librosa.audio.BW_FASTEST * nyquist:
        res_type = "kaiser_fast"
    else:
        res_type = "kaiser_best"

downsample_count1 = max(
    0, int(np.ceil(np.log2(librosa.audio.BW_FASTEST * nyquist / filter_cutoff)) - 1) - 1
)


def num_two_factors(x):
    if x <= 0:
        return 0
    num_twos = 0
    while x % 2 == 0:
        num_twos += 1
        x //= 2

    return num_twos


num_twos = num_two_factors(hop_length)
downsample_count2 = max(0, num_twos - n_octaves + 1)
downsample_count = min(downsample_count1, downsample_count2)


vqt_resp = []

num_twos = num_two_factors(hop_length)
if num_twos < n_octaves - 1:
    raise ParameterError(
        "hop_length must be a positive integer "
        "multiple of 2^{0:d} for {1:d}-octave CQT/VQT".format(n_octaves - 1, n_octaves)
    )

my_y, my_sr, my_hop = y, sr, hop_length


def sparsify_rows(x, quantile=0.01, dtype=None):
    if x.ndim == 1:
        x = x.reshape((1, -1))

    elif x.ndim > 2:
        raise ParameterError(
            "Input must have 2 or fewer dimensions. "
            "Provided x.shape={}.".format(x.shape)
        )

    if not 0.0 <= quantile < 1:
        raise ParameterError("Invalid quantile {:.2f}".format(quantile))

    if dtype is None:
        dtype = x.dtype

    x_sparse = scipy.sparse.lil_matrix(x.shape, dtype=dtype)

    mags = np.abs(x)
    norms = np.sum(mags, axis=1, keepdims=True)

    mag_sort = np.sort(mags, axis=1)
    cumulative_mag = np.cumsum(mag_sort / norms, axis=1)

    threshold_idx = np.argmin(cumulative_mag < quantile, axis=1)

    for i, j in enumerate(threshold_idx):
        idx = np.where(mags[i] >= mag_sort[i, j])
        x_sparse[i, idx] = x[i, idx]

    return x_sparse.tocsr()


def cqt_filter_fft(
    sr,
    fmin,
    n_bins,
    bins_per_octave,
    filter_scale,
    norm,
    sparsity,
    hop_length=None,
    window="hann",
    gamma=0.0,
    dtype=np.complex,
):
    basis, lengths = librosa.filters.constant_q(
        sr,
        fmin=fmin,
        n_bins=n_bins,
        bins_per_octave=bins_per_octave,
        filter_scale=filter_scale,
        norm=norm,
        pad_fft=True,
        window=window,
    )

    n_fft = basis.shape[1]

    if hop_length is not None and n_fft < 2.0 ** (1 + np.ceil(np.log2(hop_length))):
        n_fft = int(2.0 ** (1 + np.ceil(np.log2(hop_length))))

    basis *= lengths[:, np.newaxis] / float(n_fft)

    fft = librosa.get_fftlib()
    fft_basis = fft.fft(basis, n=n_fft, axis=1)[:, : (n_fft // 2) + 1]

    fft_basis = sparsify_rows(fft_basis, quantile=sparsity, dtype=dtype)

    return fft_basis, n_fft, lengths


def cqt_response(y, n_fft, hop_length, fft_basis, mode, dtype=None):
    D = librosa.stft(
        y, n_fft=n_fft, hop_length=hop_length, window="ones", pad_mode=mode, dtype=dtype
    )
    return fft_basis.dot(D)


for i in range(n_octaves):
    if i > 0:
        if len(my_y) < 2:
            raise ParameterError(
                "Input signal length={} is too short for "
                "{:d}-octave CQT/VQT".format(len_orig, n_octaves)
            )

        my_y = librosa.audio.resample(my_y, 2, 1, res_type=res_type, scale=True)

        my_sr /= 2.0
        my_hop //= 2

    fft_basis, n_fft, _ = cqt_filter_fft(
        my_sr,
        fmin_t * 2.0**-i,
        n_filters,
        bins_per_octave,
        filter_scale,
        norm,
        sparsity,
        window=window,
        gamma=gamma,
        dtype=dtype,
    )

    fft_basis[:] *= np.sqrt(2**i)

    vqt_resp.append(cqt_response(my_y, n_fft, my_hop, fft_basis, pad_mode, dtype=dtype))


def trim_stack(cqt_resp, n_bins, dtype):
    max_col = min(c_i.shape[-1] for c_i in cqt_resp)
    cqt_out = np.empty((n_bins, max_col), dtype=dtype, order="F")

    end = n_bins
    for c_i in cqt_resp:
        n_oct = c_i.shape[0]
        if end < n_oct:
            cqt_out[:end] = c_i[-end:, :max_col]
        else:
            cqt_out[end - n_oct : end] = c_i[:, :max_col]

        end -= n_oct

    return cqt_out


V = trim_stack(vqt_resp, n_bins, dtype)

if scale:
    lengths = librosa.filters.constant_q_lengths(
        sr,
        fmin,
        n_bins=n_bins,
        bins_per_octave=bins_per_octave,
        window=window,
        filter_scale=filter_scale,
    )
    V /= np.sqrt(lengths[:, np.newaxis])

test_sol = V
assert np.allclose(test_sol, sol)
