# library: matplotlib
# version: 3.8.0
# extra_dependencies: ['pyparsing==2.3.1']
import matplotlib.pyplot as plt


def use_seaborn() -> None:
    plt.style.use("seaborn-v0_8")
