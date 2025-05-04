# library: pytest
# version: 7.0.0
# extra_dependencies: []
import pytest


class CustomItem(pytest.Item):
    def __init__(self, *, additional_arg, **kwargs):
        super().__init__(**kwargs)
        self.additional_arg = additional_arg
