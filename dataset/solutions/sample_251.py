# library: falcon
# version: 2.0.0
# extra_dependencies: []
from typing import NoReturn
import falcon


def raise_too_large_error(error_message: str) -> NoReturn:
    raise falcon.HTTPPayloadTooLarge(error_message)
