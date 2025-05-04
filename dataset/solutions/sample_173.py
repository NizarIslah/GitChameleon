# library: flask
# version: 3.0.0
# extra_dependencies: ['scipy==1.11.1']
import flask
import werkzeug
from scipy import linalg
import numpy as np

error404 = werkzeug.exceptions.NotFound


def save_exponential(
    A: np.ndarray, base_path: str, sub_path: str
) -> tuple[str, np.ndarray]:
    # Attempt to join the base path and sub path.
    # If the joined path is outside the base path, raise a 404 error.
    # compute the exponential of the batched matrices (m, m) in A (n,m,m)
    # return the save_path and the exponential of the matrices

    joined = werkzeug.utils.safe_join(base_path, sub_path)
    if joined is None:
        raise error404
    output = linalg.expm(A)
    return joined, output
