# library: lightgbm
# version: 3.0.0
# extra_dependencies: ['numpy==1.26.4']
import numpy as np
import json
from lightgbm.compat import json_default_with_numpy


def dump_json(data: any) -> str:
    """
    Dump data to JSON format.

    Args:
        data (any): The data to dump.

    Returns:
        str: The JSON representation of the data.
    """
    return json.dumps(data, default=json_default_with_numpy)
