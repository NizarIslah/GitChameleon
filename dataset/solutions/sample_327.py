# library: matplotlib
# version: 3.2.0
# extra_dependencies: ['numpy==1.18.1', 'pyparsing==2.3.1', 'packaging==19.0']
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes


def modify(fig: Figure, ax: Axes) -> None:
    ax.set_xticks([], False)
    ax.set_yticks([], False)
