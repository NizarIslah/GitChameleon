# library: seaborn
# version: 0.13.0
# extra_dependencies: ['pandas==2.0.0', 'numpy==1.26.4']
import seaborn as sns
import pandas as pd
from matplotlib.axes import Axes


def custom_pointplot(data: pd.DataFrame) -> Axes:
    return sns.pointplot(x="x", y="y", data=data, markers="o", linestyles="none")
