# library: scikit-learn
# version: 1.3
# extra_dependencies: ['numpy==1.23.5']
from sklearn.datasets import load_digits
from sklearn.decomposition import FastICA
from sklearn.utils import Bunch


def apply_fast_ica(data: Bunch, n_components: int) -> FastICA:
    return FastICA(
        n_components=n_components, random_state=0, whiten="arbitrary-variance"
    ).fit_transform(data)
