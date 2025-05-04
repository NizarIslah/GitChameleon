# library: nltk
# version: 3.5
# extra_dependencies: []
from nltk.lm.api import accumulate
import operator


def accumulate_functional(iterable, func):
    """
    Accumulate the results of applying a function to an iterable.

    Args:
        iterable (iterable): An iterable to accumulate.
        func (function): A function to apply to the elements of the iterable.

    Returns:
        list: A list of accumulated results.
    """
    return list(accumulate(iterable, func))
