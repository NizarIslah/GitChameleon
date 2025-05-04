# library: falcon
# version: 3.0.0
# extra_dependencies: []
import falcon
from falcon import HTTPStatus


def custom_body(status: falcon.HTTPStatus, info: str) -> falcon.HTTPStatus:
    status.text = info
    return status
