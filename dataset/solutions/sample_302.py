# library: librosa
# version: 0.7.1
# extra_dependencies: ['pip==24.0', 'numba==0.46', 'llvmlite==0.30', 'joblib==0.12', 'numpy==1.16.0', 'audioread==2.1.5', 'scipy==1.1.0', 'resampy==0.2.2', 'soundfile==0.10.2']
import librosa
import numpy as np


def compute_shear(E: np.ndarray, factor: int, axis: int) -> np.ndarray:
    return librosa.util.shear(E, factor=factor, axis=axis)
