# library: librosa
# version: 0.6.0
# extra_dependencies: ['pip==24.0', 'numba==0.46', 'llvmlite==0.30', 'joblib==0.12', 'numpy==1.16.0', 'audioread==2.1.5', 'scipy==1.1.0', 'resampy==0.2.2']
import librosa
import numpy as np


def compute_rms(y: np.ndarray) -> np.float32:
    return librosa.feature.rmse(y=y)
