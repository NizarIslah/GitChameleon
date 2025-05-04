# library: pytest
# version: 7.0.0
# extra_dependencies: []
import pytest


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_setup():
    yield
