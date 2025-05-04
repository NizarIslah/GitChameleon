# library: flask
# version: 2.0.1
# extra_dependencies: ['werkzeug==2.0.0']
import flask
import werkzeug

error404 = werkzeug.exceptions.NotFound


def safe_join_fail_404(base_path: str, sub_path: str) -> str:
    # Attempt to join the base path and sub path.
    # If the joined path is outside the base path, raise a 404 error.

    joined = flask.safe_join(base_path, sub_path)

    return joined
