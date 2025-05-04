# library: pytest
# version: 7.0.0
# extra_dependencies: []
import pytest
import pathlib


@pytest.hookimpl()
def pytest_collect_file(file_path: pathlib.Path):
    pass
