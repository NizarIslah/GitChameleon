import unittest
import sys
import os
from typing import List
import numpy as np

# Add the parent directory to sys.path to import the module to test
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sample_175 import custom_generateRandomSampleDice

# Import required libraries from sample_175.py
from sympy.stats import Die
import sympy.stats.rv

dice = Die("X", 6)
import warnings
from sympy.utilities.exceptions import SymPyDeprecationWarning


def test_custom_generateRandomSampleDice():
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always", SymPyDeprecationWarning)  # Capture all warnings
        output = custom_generateRandomSampleDice(dice, 3)
        expect = [sample(dice) for i in range(3)]
        expect = [sample(dice) for i in range(3)]
        assert isinstance(output, list), "Test Failed: Output is not a list!"
        assert len(output) == len(
            expect
        ), "Test Failed: Output length does not match expected!"
        assert not any(
            isinstance(warn.message, SymPyDeprecationWarning) for warn in w
        ), "Test Failed: Deprecation warning was triggered!"


test_custom_generateRandomSampleDice()
