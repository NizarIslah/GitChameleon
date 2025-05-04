# library: pytest
# version: 7.2.0
# extra_dependencies: []
import pytest


def foo(a, b):
    return (10 * a - b + 7) // 3


@pytest.mark.parametrize(
    ["a", "b", "result"],
    [
        [1, 2, 5],
        [2, 3, 8],
        [5, 3, 18],
    ],
)
def test_foo(a: int, b: int, result: int) -> None:
    assert foo(a, b) == result
