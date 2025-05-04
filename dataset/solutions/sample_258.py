# library: tornado
# version: 6.3.0
# extra_dependencies: []
import asyncio
import os
import signal
from typing import Callable


def custom_add_callback_from_signal(callback: Callable[[], None], signum: int) -> None:
    loop = asyncio.get_event_loop()
    loop.add_signal_handler(signum, callback)
