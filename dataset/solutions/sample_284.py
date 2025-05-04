# library: librosa
# version: 0.7.0
# extra_dependencies: ['pip==24.0', 'scikit-learn==0.21.0', 'numba==0.46', 'llvmlite==0.30', 'joblib==0.12', 'numpy==1.16.0', 'audioread==2.1.5', 'scipy==1.1.0', 'resampy==0.2.2']
import librosa
import numpy as np


# Save the stream in variable stream. Save each stream block with the array stream_blocks
def compute_stream(y, sr, n_fft, hop_length):
    stream_blocks = []

    stream = librosa.stream(
        filename,
        block_length=16,
        frame_length=n_fft,
        hop_length=hop_length,
        mono=True,
        fill_value=0,
    )

    for c, y_block in enumerate(stream):
        stream_blocks.append(
            librosa.stft(y_block, n_fft=n_fft, hop_length=hop_length, center=False)
        )
    return stream, stream_blocks
