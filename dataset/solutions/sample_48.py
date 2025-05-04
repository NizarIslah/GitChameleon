# library: scikit-learn
# version: 1.1
# extra_dependencies: ['numpy==1.23.5']
from sklearn.datasets import load_digits
from sklearn.utils import Bunch
from sklearn.decomposition import FastICA


def apply_fast_ica(data: Bunch, n_components: int) -> FastICA:
    return FastICA(
        n_components=n_components, random_state=0, whiten=True
    ).fit_transform(data)
