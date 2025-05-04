import os
import sys
import unittest

import tornado.testing

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from sample_263 import DummyAuth


async def custom_auth_test():
    auth = DummyAuth()
    result = await auth.async_get_user_info("dummy_token")
    expect = "dummy_token"
    assert result["token"] == expect


async def main():
    result = await custom_auth_test()


if __name__ == "__main__":
    asyncio.run(main())
