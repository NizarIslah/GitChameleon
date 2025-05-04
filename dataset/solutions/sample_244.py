# library: falcon
# version: 3.0.0
# extra_dependencies: []
from falcon.stream import BoundedStream


def custom_writable(bstream: BoundedStream) -> bool:
    return bstream.writable()
