# library: lightgbm
# version: 4.3.0
# extra_dependencies: ['numpy==1.26.4']
import lightgbm as lgb
import numpy as np


def convert_from_sliced_object(sliced_data: np.ndarray) -> np.ndarray:
    """
    Convert a sliced object to a fixed object.

    Args:
        sliced_data (np.ndarray): The sliced object to convert.

    Returns:
        np.ndarray: The converted fixed object.
    """
    return lgb.basic._convert_from_sliced_object(sliced_data)
