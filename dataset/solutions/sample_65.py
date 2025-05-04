# library: pandas
# version: 2
# extra_dependencies: ['numpy==1.21.6']
import pandas as pd


def combined(
    df1: pd.DataFrame, df2: pd.DataFrame, series1: pd.Series, series2: pd.Series
) -> tuple:
    return pd.concat([df1, df2], ignore_index=True), pd.concat(
        [series1, series2], ignore_index=True
    )
