# library: scipy
# version: 1.11.2
# extra_dependencies: []

from scipy.stats import rv_continuous


def compute_moment(dist: rv_continuous, n: int) -> float:
    return dist.moment(order=n)
