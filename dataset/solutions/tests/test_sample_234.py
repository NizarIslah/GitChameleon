import pytest
import sys
import os

# Add the parent directory to sys.path to import the module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from sample_234 import foo


class TestFoo:
    """Test class for the foo function in sample_234.py"""

    @pytest.mark.parametrize(
        ["a", "b", "expected"],
        [
            (1, 2, 5),
            (2, 3, 8),
            (5, 3, 18),
            (0, 0, 2),  # Edge case: (10*0 - 0 + 7) // 3 = 7 // 3 = 2
            (10, 20, 29),  # Additional test case
            (
                -1,
                -2,
                -1,
            ),  # Negative numbers: (10*(-1) - (-2) + 7) // 3 = (-10 + 2 + 7) // 3 = -1 // 3 = -1
        ],
    )
    def test_foo_calculation(self, a: int, b: int, expected: int) -> None:
        """Test that foo function correctly calculates (10 * a - b + 7) // 3"""
        assert foo(a, b) == expected

    def test_foo_types(self) -> None:
        """Test that foo function handles different numeric types correctly"""
        # Test with float inputs: (10 * 1.5 - 2.5 + 7) // 3 => 19.5 // 3 => 6.0, which is equal to 6 in Python
        assert foo(1.5, 2.5) == 6

        # Test with string inputs (should raise TypeError)
        with pytest.raises(TypeError):
            foo("1", "2")

    def test_foo_formula(self) -> None:
        """Test that foo function follows the formula (10 * a - b + 7) // 3"""
        a, b = 3, 4
        expected = (10 * a - b + 7) // 3
        assert foo(a, b) == expected
