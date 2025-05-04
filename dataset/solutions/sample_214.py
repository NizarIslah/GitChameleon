# library: seaborn
# version: 0.12.0
# extra_dependencies: ['pandas==1.4.0', 'numpy==1.26.4']
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.axes import Axes


def custom_set_axis_labels(data: pd.DataFrame) -> Axes:
    ax = sns.scatterplot(x="x", y="y", data=data)
    ax.set(xlabel="My X Label", ylabel="My Y Label")
    return ax
