# library: jinja2
# version: 3.1
# extra_dependencies: []
import jinja2
from jinja2.runtime import Context
from typing import Callable


def setup_environment(filtername: str, filter) -> jinja2.Environment:
    env = jinja2.Environment()
    env.filters[filtername] = filter
    return env


def solution() -> Callable[[Context, str], str]:
    @jinja2.pass_context
    def greet(ctx, name):
        prefix = ctx.get("prefix", "Hello")
        return f"{prefix}, {name}!"

    return greet
