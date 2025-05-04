# library: pandas
# version: 1.4.0
# extra_dependencies: ['numpy==1.21.6']
import pandas as pd


def correct_type(index: pd.Index) -> str:
    return "int64"
