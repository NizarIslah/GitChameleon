# library: falcon
# version: 3.0.0
# extra_dependencies: []
from falcon import Response
import falcon


def custom_append_link(resp: falcon.Response, link: str, rel: str) -> falcon.Response:
    resp.append_link(link, rel, crossorigin="anonymous")
    return resp
