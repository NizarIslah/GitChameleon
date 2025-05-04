# library: librosa
# version: 0.6.0
# extra_dependencies: ['pip==24.0', 'scikit-learn==0.21.0', 'numba==0.46', 'llvmlite==0.30', 'joblib==0.12', 'numpy==1.16.0', 'audioread==2.1.5', 'scipy==1.1.0', 'resampy==0.2.2', 'soundfile==0.10.2']
import librosa
import numpy as np
import soundfile as sf


# Save the stream in variable stream. Save each stream block with the array stream_blocks
def compute_stream(filename, y, sr, n_fft, hop_length):
    stream_blocks = []

    stream = sf.blocks(
        filename,
        blocksize=n_fft + 15 * hop_length,
        overlap=n_fft - hop_length,
        fill_value=0,
    )

    for c, block in enumerate(stream):
        y = librosa.to_mono(block.T)
        D = librosa.stft(y, n_fft=n_fft, hop_length=hop_length, center=False)
        stream_blocks.append(D)

    return stream, stream_blocks
