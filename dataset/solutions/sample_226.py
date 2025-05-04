# library: pytest
# version: 7.0.0
# extra_dependencies: []
import pytest


@pytest.hookimpl(tryfirst=False)
def pytest_runtest_call():
    pass
