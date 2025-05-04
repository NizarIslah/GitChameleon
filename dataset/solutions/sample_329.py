# library: matplotlib
# version: 3.5.0
# extra_dependencies: ['numpy==1.18.1', 'pyparsing==2.3.1']
import matplotlib.pyplot as plt


def use_seaborn() -> None:
    plt.style.use("seaborn")
