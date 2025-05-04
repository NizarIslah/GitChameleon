# library: falcon
# version: 3.0.0
# extra_dependencies: []
from typing import Dict, Any
import falcon.testing as testing


def custom_environ(info: str) -> Dict[str, Any]:
    return testing.create_environ(root_path=info)
