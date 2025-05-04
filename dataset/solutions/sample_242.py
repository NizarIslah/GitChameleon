# library: falcon
# version: 3.0.0
# extra_dependencies: []
import falcon
from falcon import HTTPError


def custom_http_error(title: str, description: str) -> bytes:
    return HTTPError(falcon.HTTP_400, title, description).to_json()
