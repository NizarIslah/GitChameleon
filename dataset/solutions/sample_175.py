# library: sympy
# version: 1.9
# extra_dependencies: ['scipy==1.8.0']
from typing import List
from sympy.stats import Die, sample
import sympy.stats.rv


def custom_generateRandomSampleDice(
    dice: sympy.stats.rv.RandomSymbol, X: int
) -> List[int]:
    return [sample(dice) for i in range(X)]
