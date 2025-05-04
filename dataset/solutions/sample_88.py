# library: lightgbm
# version: 4.3.0
# extra_dependencies: []
import ctypes
import lightgbm.basic as basic


def create_c_array(values: list, ctype: type) -> ctypes.Array:
    """
    Create a ctypes array from a list of values.
    Args:
        values (list): A list of values to be converted to a ctypes array.
        ctype (type): The ctypes type of the array elements.
    Returns:
        ctypes.Array: A ctypes array containing the values.
    """
    return basic._c_array(ctype, values)
