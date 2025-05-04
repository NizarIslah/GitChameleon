# library: flask
# version: 2.0.1
# extra_dependencies: ['werkzeug==2.0.0']
import flask
import datetime


def convert_timedelta_to_seconds(td: datetime.timedelta) -> int:
    return flask.helpers.total_seconds(td)
