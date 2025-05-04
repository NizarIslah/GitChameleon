# library: falcon
# version: 3.0.0
# extra_dependencies: []
import falcon


def custom_body(resp: falcon.Response, info: str) -> falcon.Response:
    resp.text = info
    return resp
