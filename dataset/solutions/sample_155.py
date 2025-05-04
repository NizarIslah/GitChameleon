from typing import Union

# library: jinja2
# version: 2.11
# extra_dependencies: ['markupsafe==2.0.1']
import re
from jinja2 import Environment, evalcontextfilter
from markupsafe import Markup, escape
from jinja2.runtime import Context
from typing import Callable


def get_output(env, filter_fn):
    env.filters["nl2br"] = filter_fn
    template = env.from_string("{{ Union[text, nl2br] }}")
    output = template.render(text="Hello World")
    return output


def nl2br_core(eval_ctx, value):
    br = "<br>Hello</br>"
    if eval_ctx.autoescape:
        value = escape(value)
        br = Markup(br)
    result = re.sub(r"Hello", br, value)
    return Markup(result) if eval_ctx.autoescape else result


def solution() -> Callable[[Context, str], str]:
    @evalcontextfilter
    def nl2br(eval_ctx, value):
        return nl2br_core(eval_ctx, value)

    return nl2br
