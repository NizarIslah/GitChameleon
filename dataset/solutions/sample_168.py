# library: flask
# version: 3.0.0
# extra_dependencies: ['numpy==1.25.1']
import flask
import werkzeug
import numpy as np

error404 = werkzeug.exceptions.NotFound


def stack_and_save(
    arr_list: list[np.ndarray],
    base_path: str,
    sub_path: str,
    casting_policy: str,
    out_dtype: type,
) -> tuple[str, np.ndarray]:
    # Attempt to join the base path and sub path.
    # If the joined path is outside the base path, raise a 404 error.
    # stack the arrays in arr_list with the casting policy and the out_dtype.
    # if the out_dtype is not compatible with the casting policy, raise a TypeError
    # and out_dtype could be np.float32 or np.float64
    # casting policy could be safe or unsafe
    # Return the joined path and the stacked array to be saved
    joined = werkzeug.utils.safe_join(base_path, sub_path)
    if joined is None:
        raise error404
    stacked = np.vstack(arr_list, casting=casting_policy, dtype=out_dtype)
    return joined, stacked
