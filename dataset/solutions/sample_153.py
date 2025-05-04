# library: jinja2
# version: 2.11
# extra_dependencies: ['markupsafe==2.0.1']
import jinja2
from jinja2.runtime import Context
from typing import Callable


def setup_environment(
    filtername: str, filter: Callable[[Context, str], str]
) -> jinja2.Environment:
    env = jinja2.Environment()
    env.filters[filtername] = filter
    return env


def solution() -> Callable[[Context, str], str]:
    @jinja2.contextfilter
    def greet(ctx, name):
        prefix = ctx.get("prefix", "Hello")
        return f"{prefix}, {name}!"

    return greet
