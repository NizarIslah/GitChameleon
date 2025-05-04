#!/usr/bin/env python3
# Test file for sample_287.py

import os

# Add the parent directory to the path so we can import the sample module
import sys
import unittest

import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sample_287 import compute_lpc_coef


filename = librosa.util.example_audio_file()
y, sr = librosa.load(filename)
order = 2

sol = compute_lpc_coef(y, sr, order)
dtype = y.dtype.type
ar_coeffs = np.zeros(order + 1, dtype=dtype)
ar_coeffs[0] = dtype(1)
ar_coeffs_prev = np.zeros(order + 1, dtype=dtype)
ar_coeffs_prev[0] = dtype(1)
fwd_pred_error = y[1:]
bwd_pred_error = y[:-1]
den = np.dot(fwd_pred_error, fwd_pred_error) + np.dot(bwd_pred_error, bwd_pred_error)
for i in range(order):
    if den <= 0:
        raise FloatingPointError("numerical error, input ill-conditioned?")
    reflect_coeff = dtype(-2) * np.dot(bwd_pred_error, fwd_pred_error) / dtype(den)
    ar_coeffs_prev, ar_coeffs = ar_coeffs, ar_coeffs_prev
    for j in range(1, i + 2):
        ar_coeffs[j] = ar_coeffs_prev[j] + reflect_coeff * ar_coeffs_prev[i - j + 1]
    fwd_pred_error_tmp = fwd_pred_error
    fwd_pred_error = fwd_pred_error + reflect_coeff * bwd_pred_error
    bwd_pred_error = bwd_pred_error + reflect_coeff * fwd_pred_error_tmp
    q = dtype(1) - reflect_coeff**2
    den = q * den - bwd_pred_error[-1] ** 2 - fwd_pred_error[0] ** 2
    fwd_pred_error = fwd_pred_error[1:]
    bwd_pred_error = bwd_pred_error[:-1]

test_sol = ar_coeffs
assert np.array_equal(test_sol, sol)
