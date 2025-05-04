# library: falcon
# version: 3.0.0
# extra_dependencies: []
from falcon import Response
import falcon


def custom_data(resp: falcon.Response, info: str) -> str:
    resp.data = info
    return resp.render_body()
