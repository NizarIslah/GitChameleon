# library: falcon
# version: 3.0.0
# extra_dependencies: []
from falcon import stream

import io


class DummyRequest:
    def __init__(self, data: bytes):
        self.stream = io.BytesIO(data)
        self.content_length = len(data)


def get_bounded_stream(req: DummyRequest) -> stream.BoundedStream:
    return stream.BoundedStream(req.stream, req.content_length)
