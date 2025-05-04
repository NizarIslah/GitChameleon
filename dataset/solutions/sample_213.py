# library: seaborn
# version: 0.13.0
# extra_dependencies: ['pandas==2.0.0', 'numpy==1.26.4']
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.axes import Axes


def custom_boxenplot(data: pd.DataFrame) -> Axes:
    return sns.boxenplot(x="x", y="y", data=data, width_method="exponential")
