# library: falcon
# version: 2.0.0
# extra_dependencies: []
class CustomRouter:
    def __init__(self):
        self.routes = {}


def solution() -> None:
    def add_route(self, uri_template, resource, **kwargs):
        from falcon.routing import map_http_methods

        method_map = map_http_methods(resource, kwargs.get("fallback", None))
        self.routes[uri_template] = (resource, method_map)
        return method_map

    CustomRouter.add_route = add_route
