# library: flask
# version: 2.0.0
# extra_dependencies: ['scipy==1.8.1', 'Werkzeug==2.0.0']
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

    joined = flask.safe_join(base_path, sub_path)
    output = np.zeros(A.shape)
    for i in range(A.shape[0]):
        output[i] = linalg.expm(A[i])
    return joined, output
