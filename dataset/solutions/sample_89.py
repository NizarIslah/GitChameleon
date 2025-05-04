# library: lightgbm
# version: 4.3.0
# extra_dependencies: []
import lightgbm as lgb
import ctypes


def c_str(python_string: str) -> ctypes.c_char_p:
    """
    Convert a Python string to a ctypes c_char_p.

    Args:
        python_string (str): The Python string to convert.

    Returns:
        ctypes.c_char_p: The converted ctypes c_char_p.
    """
    return lgb.basic._c_str(python_string)
