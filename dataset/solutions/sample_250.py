# library: falcon
# version: 3.0.0
# extra_dependencies: []
import json
from falcon import Request
from falcon.testing import create_environ


def custom_media(req: Request) -> dict[str, str]:
    return req.get_media()
