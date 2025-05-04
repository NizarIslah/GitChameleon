# library: falcon
# version: 3.0.0
# extra_dependencies: []
from falcon import Response
import falcon


def custom_link(resp: Response, link_rel: str, link_href: str) -> falcon.Response:
    resp.append_link(link_href, link_rel)
    return resp
