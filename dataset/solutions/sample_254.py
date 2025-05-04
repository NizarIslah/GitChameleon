# library: falcon
# version: 2.0.0
# extra_dependencies: []
import falcon
import logging
from typing import Any, Dict


def handle_error(
    req: falcon.Request, resp: falcon.Response, ex: Exception, params: Dict[str, Any]
) -> None:
    req_path = getattr(req, "path", "unknown")
    resp.media = {
        "error": str(ex),
        "details": {
            "request": req_path,
            "params": params,
        },
    }
    resp.status = falcon.HTTP_500
