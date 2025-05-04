# library: librosa
# version: 0.7.0
# extra_dependencies: ['pip==24.0', 'numba==0.46', 'llvmlite==0.30', 'joblib==0.14', 'numpy==1.16.0', 'audioread==2.1.5', 'scipy==1.1.0', 'resampy==0.2.2', 'soundfile==0.10.2']
import librosa
import numpy as np
import scipy
from typing import Union

DTypeLike = Union[np.dtype, type]


def compute_vqt(
    y: np.ndarray,
    sr: int,
    hop_length: int,
    fmin: int,
    n_bins: int,
    gamma: int,
    bins_per_octave: int,
    tuning: float,
    filter_scale: int,
    norm: 1,
    sparsity: float,
    window: str,
    scale: bool,
    pad_mode: str,
    res_type: str,
    dtype: DTypeLike,
) -> np.ndarray:
    # How many octaves are we dealing with?
    def dtype_r2c(d, default=np.complex64):
        """Find the complex numpy dtype corresponding to a real dtype.

        This is used to maintain numerical precision and memory footprint
        when constructing complex arrays from real-valued data
        (e.g. in a Fourier transform).

        A `float32` (single-precision) type maps to `complex64`,
        while a `float64` (double-precision) maps to `complex128`.


        Parameters
        ----------
        d : np.dtype
            The real-valued dtype to convert to complex.
            If ``d`` is a complex type already, it will be returned.

        default : np.dtype, optional
            The default complex target type, if ``d`` does not match a
            known dtype

        Returns
        -------
        d_c : np.dtype
            The complex dtype

        See Also
        --------
        dtype_c2r
        numpy.dtype

        """
        mapping = {
            np.dtype(np.float32): np.complex64,
            np.dtype(np.float64): np.complex128,
            np.dtype(np.float): np.complex,
        }

        # If we're given a complex type already, return it
        dt = np.dtype(d)
        if dt.kind == "c":
            return dt

        # Otherwise, try to map the dtype.
        # If no match is found, return the default.
        return np.dtype(mapping.get(dt, default))

    n_octaves = int(np.ceil(float(n_bins) / bins_per_octave))
    n_filters = min(bins_per_octave, n_bins)

    len_orig = len(y)

    # Relative difference in frequency between any two consecutive bands
    alpha = 2.0 ** (1.0 / bins_per_octave) - 1

    if fmin is None:
        # C1 by default
        fmin = librosa.note_to_hz("C1")

    if tuning is None:
        tuning = librosa.pitch.estimate_tuning(
            y=y, sr=sr, bins_per_octave=bins_per_octave
        )

    if gamma is None:
        gamma = 24.7 * alpha / 0.108

    if dtype is None:
        dtype = dtype_r2c(y.dtype)

    # Apply tuning correction
    fmin = fmin * 2.0 ** (tuning / bins_per_octave)

    # First thing, get the freqs of the top octave
    freqs = librosa.time_frequency.cqt_frequencies(
        n_bins, fmin, bins_per_octave=bins_per_octave
    )[-bins_per_octave:]

    fmin_t = np.min(freqs)
    fmax_t = np.max(freqs)

    # Determine required resampling quality
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
        0,
        int(np.ceil(np.log2(librosa.audio.BW_FASTEST * nyquist / filter_cutoff)) - 1)
        - 1,
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

    # Make sure our hop is long enough to support the bottom octave

    num_twos = num_two_factors(hop_length)

    # num_twos = __num_two_factors(hop_length)
    if num_twos < n_octaves - 1:
        raise ParameterError(
            "hop_length must be a positive integer "
            "multiple of 2^{0:d} for {1:d}-octave CQT/VQT".format(
                n_octaves - 1, n_octaves
            )
        )

    # Now do the recursive bit
    my_y, my_sr, my_hop = y, sr, hop_length

    def sparsify_rows(x, quantile=0.01, dtype=None):
        """Return a row-sparse matrix approximating the input

        Parameters
        ----------
        x : np.ndarray [ndim <= 2]
            The input matrix to sparsify.

        quantile : float in [0, 1.0)
            Percentage of magnitude to discard in each row of ``x``

        dtype : np.dtype, optional
            The dtype of the output array.
            If not provided, then ``x.dtype`` will be used.

        Returns
        -------
        x_sparse : ``scipy.sparse.csr_matrix`` [shape=x.shape]
            Row-sparsified approximation of ``x``

            If ``x.ndim == 1``, then ``x`` is interpreted as a row vector,
            and ``x_sparse.shape == (1, len(x))``.

        Raises
        ------
        ParameterError
            If ``x.ndim > 2``

            If ``quantile`` lies outside ``[0, 1.0)``
        """

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
        """Generate the frequency domain constant-Q filter basis."""

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

        # Filters are padded up to the nearest integral power of 2
        n_fft = basis.shape[1]

        if hop_length is not None and n_fft < 2.0 ** (1 + np.ceil(np.log2(hop_length))):
            n_fft = int(2.0 ** (1 + np.ceil(np.log2(hop_length))))

        # re-normalize bases with respect to the FFT window length
        basis *= lengths[:, np.newaxis] / float(n_fft)

        # FFT and retain only the non-negative frequencies
        fft = librosa.get_fftlib()
        fft_basis = fft.fft(basis, n=n_fft, axis=1)[:, : (n_fft // 2) + 1]

        # sparsify the basis
        fft_basis = sparsify_rows(fft_basis, quantile=sparsity, dtype=dtype)

        return fft_basis, n_fft, lengths

    def cqt_response(y, n_fft, hop_length, fft_basis, mode, dtype=None):
        """Compute the filter response with a target STFT hop."""

        # Compute the STFT matrix
        D = librosa.stft(
            y,
            n_fft=n_fft,
            hop_length=hop_length,
            window="ones",
            pad_mode=mode,
            dtype=dtype,
        )

        # And filter response energy
        return fft_basis.dot(D)

    # Iterate down the octaves
    for i in range(n_octaves):
        # Resample (except first time)
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

        # Re-scale the filters to compensate for downsampling
        fft_basis[:] *= np.sqrt(2**i)

        # Compute the vqt filter response and append to the stack
        vqt_resp.append(
            cqt_response(my_y, n_fft, my_hop, fft_basis, pad_mode, dtype=dtype)
        )

    def trim_stack(cqt_resp, n_bins, dtype):
        """Helper function to trim and stack a collection of CQT responses"""

        max_col = min(c_i.shape[-1] for c_i in cqt_resp)
        cqt_out = np.empty((n_bins, max_col), dtype=dtype, order="F")

        # Copy per-octave data into output array
        end = n_bins
        for c_i in cqt_resp:
            # By default, take the whole octave
            n_oct = c_i.shape[0]
            # If the whole octave is more than we can fit,
            # take the highest bins from c_i
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
    return V
