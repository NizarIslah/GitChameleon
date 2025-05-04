# library: lightgbm
# version: 3.0.0
# extra_dependencies: ['numpy==1.26.4']
import lightgbm as lgb
import numpy as np
import ctypes


def convert_cint32_array_to_numpy(c_pointer: ctypes.POINTER, length: int) -> np.ndarray:
    """
    Convert a ctypes pointer to a numpy array.

    Args:
        c_pointer (c_array_type): A ctypes pointer to an array of integers.
        length (int): The length of the array.

    Returns:
        np.ndarray: A numpy array containing the elements of the ctypes array.
    """
    return lgb.basic.cint32_array_to_numpy(c_pointer, length)
