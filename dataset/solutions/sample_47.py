# library: scikit-learn
# version: 1.1
# extra_dependencies: ['numpy==1.23.5']
from sklearn.datasets import make_sparse_coded_signal


def get_signal(
    n_samples: int, n_features: int, n_components: int, n_nonzero_coefs: int
) -> tuple:
    return make_sparse_coded_signal(
        n_samples=n_samples,
        n_features=n_features,
        n_components=n_components,
        n_nonzero_coefs=n_nonzero_coefs,
    )
