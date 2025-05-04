# library: mitmproxy
# version: 7.0.0
# extra_dependencies: []
from mitmproxy.http import Headers


def custom_function(header_name: bytes, initial_value: bytes) -> Headers:
    return Headers([(header_name, initial_value)])
