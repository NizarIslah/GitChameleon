# library: falcon
# version: 3.0.0
# extra_dependencies: []
import falcon.app_helpers as app_helpers


class ExampleMiddleware:
    def process_request(self, req, resp):
        pass


def custom_middleware_variable() -> list[ExampleMiddleware]:
    return [ExampleMiddleware()]
