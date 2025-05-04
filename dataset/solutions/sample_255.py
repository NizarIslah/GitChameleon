# library: falcon
# version: 2.0.0
# extra_dependencies: []
from falcon import Request


def custom_get_dpr(req: Request) -> int:
    return req.get_param_as_int("dpr", min_value=0, max_value=3)
