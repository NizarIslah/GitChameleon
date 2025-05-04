# library: pandas
# version: 1.5.0
# extra_dependencies: ['numpy==1.21.6']
import pandas as pd


def get_grouped_df(df: pd.DataFrame) -> pd.DataFrame:
    return df.groupby("x", observed=False, dropna=False).sum()
