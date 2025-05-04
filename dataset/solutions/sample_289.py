# library: librosa
# version: 0.6.0
# extra_dependencies: ['pip==24.0', 'scikit-learn==0.21.0', 'numba==0.46', 'llvmlite==0.30', 'joblib==0.12', 'numpy==1.16.0', 'audioread==2.1.5', 'scipy==1.1.0', 'resampy==0.2.2', 'soundfile==0.10.2']
import librosa
import numpy as np
from librosa.core.spectrum import stft


def compute_fourier_tempogram(oenv: np.ndarray, sr: int, hop_length: int) -> np.ndarray:
    """
    Compute the Fourier tempogram: the short-time Fourier transform of the onset strength envelope.

    Parameters:
       oenv: The onset strength envelope.
       sr: The sampling rate of the audio signal in Hertz.
       hop_length: The number of samples between successive frames.

    Returns:
       The computed Fourier tempogram.
    """

    return stft(oenv, n_fft=384, hop_length=1, center=True, window="hann")
