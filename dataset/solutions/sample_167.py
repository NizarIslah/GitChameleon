# library: flask
# version: 2.0.0
# extra_dependencies: ['numpy==1.21.6', 'werkzeug==2.0.0']
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
    joined = flask.safe_join(base_path, sub_path)
    casted_list = []

    for arr in arr_list:
        if not np.can_cast(arr.dtype, out_dtype, casting=casting_policy):
            raise TypeError("Cannot cast array")
        casted_list.append(arr.astype(out_dtype, copy=False))

    stacked = np.vstack(casted_list)
    return joined, stacked
