# library: lightgbm
# version: 3.0.0
# extra_dependencies: ['numpy==1.26.4']
import lightgbm as lgb
import numpy as np


def get_params(dataset: lgb.Dataset) -> dict:
    """
    Get the parameters of the dataset.

    Args:
        dataset (lgb.Dataset): The dataset to get the parameters from.

    Returns:
        dict: The parameters of the dataset.
    """
    return dataset.get_params()
