# library: nltk
# version: 3.6.3
# extra_dependencies: []
import nltk
import io
import contextlib


def show_usage(obj: object) -> str:
    with io.StringIO() as buf, contextlib.redirect_stdout(buf):
        nltk.usage(obj)
        return buf.getvalue()
