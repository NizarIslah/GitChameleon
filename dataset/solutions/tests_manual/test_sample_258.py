import asyncio
import signal

# Import the function to test
import sys
import unittest
from unittest.mock import MagicMock, patch

sys.path.append("/repo/dataset/solutions")
from sample_258 import custom_add_callback_from_signal


def test_custom_signal_handler():
    flag = {"executed": False}
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    def callback():
        flag["executed"] = True
        loop.stop()

    custom_add_callback_from_signal(callback, signal.SIGUSR1)

    os.kill(os.getpid(), signal.SIGUSR1)

    loop.run_forever()

    return flag["executed"]


result = test_custom_signal_handler()
assert result
