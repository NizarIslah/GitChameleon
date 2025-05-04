# library: librosa
# version: 0.7.0
# extra_dependencies: ['pip==24.0', 'numba==0.46', 'llvmlite==0.30', 'joblib==0.12', 'numpy==1.16.0', 'audioread==2.1.5', 'scipy==1.1.0', 'resampy==0.2.2', 'soundfile==0.10.2']
import librosa
import numpy as np
from librosa.core.spectrum import stft, istft
from typing import Optional


def compute_plp(
    y: np.ndarray,
    sr: int,
    hop_length: int,
    win_length: int,
    tempo_min: Optional[float],
    tempo_max: Optional[float],
    onset_env: np.ndarray,
) -> np.ndarray:
    """
    Compute the Predominant Local Pulse (PLP) of an audio signal.

    Parameters:
        y: The audio signal.
        sr: The sampling rate of the audio signal in Hertz.
        hop_length: The number of samples between successive frames.
        win_length: The length (in samples) of the analysis window.
        tempo_min: The minimum tempo (in BPM) for consideration.
        tempo_max: The maximum tempo (in BPM) for consideration.
        onset_env: The onset envelope of the audio signal.

    Returns:
        The computed PLP (Predominant Local Pulse) values.
    """

    return librosa.beat.plp(
        onset_envelope=onset_env, sr=sr, tempo_min=tempo_min, tempo_max=tempo_max
    )
