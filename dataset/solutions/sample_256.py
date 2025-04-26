# library: falcon
# version: 2.0.0
# extra_dependencies: []
from falcon import Request
from falcon.util.structures import Context


def custom_set_context(req: Request, role: str, user: str) -> Context:
    req.context.role = role
    req.context.user = user
    return req.context
