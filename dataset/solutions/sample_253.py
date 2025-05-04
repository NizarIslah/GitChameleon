# library: falcon
# version: 2.0.0
# extra_dependencies: []
from falcon import Request


def custom_get_param(req: Request) -> dict[str, str]:
    return req.get_param_as_json("foo")
