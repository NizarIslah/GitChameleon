# library: falcon
# version: 3.0.0
# extra_dependencies: []
from falcon import Response


def custom_body_length(resp: Response, info):
    resp.content_length = len(info)
    return resp
