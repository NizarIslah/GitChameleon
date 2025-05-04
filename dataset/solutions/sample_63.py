# library: pandas
# version: 1.4.0
# extra_dependencies: ['numpy==1.21.6']
import pandas as pd


def combined(
    df1: pd.DataFrame, df2: pd.DataFrame, series1: pd.Series, series2: pd.Series
) -> tuple:
    return df1.append(df2, ignore_index=True), series1.append(
        series2, ignore_index=True
    )
