# library: tornado
# version: 6.0.0
# extra_dependencies: []
import asyncio
import tornado.auth
import asyncio


class DummyAuth(tornado.auth.OAuth2Mixin):
    async def async_get_user_info(self, access_token: str) -> dict[str, str]:
        return {"user": "test", "token": access_token}
