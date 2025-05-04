# library: sympy
# version: 1.9
# extra_dependencies: []
from sympy import Indexed, Symbol
import sympy
from typing import Set


def custom_symbol(index: Indexed) -> set[Symbol]:
    return index.free_symbols
