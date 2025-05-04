# library: pytest
# version: 7.0.0
# extra_dependencies: []
import pytest
import pathlib


@pytest.hookimpl()
def pytest_report_header(start_path: pathlib.Path):
    pass
