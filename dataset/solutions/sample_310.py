# library: librosa
# version: 0.8.0
# extra_dependencies: ['pip==24.0', 'numba==0.46', 'llvmlite==0.30', 'joblib==0.14', 'numpy==1.16.0', 'audioread==2.1.5', 'scipy==1.1.0', 'resampy==0.2.2', 'soundfile==0.10.2']
import librosa
import numpy as np
import scipy
from typing import Union, Optional

DTypeLike = Union[np.dtype, type]


def compute_vqt(y: np.ndarray, sr: int) -> np.ndarray:
    return librosa.vqt(y, sr=sr)
