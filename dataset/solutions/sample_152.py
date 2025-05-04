# library: flask
# version: 3.0.0
# extra_dependencies: []
import flask
import datetime


def convert_timedelta_to_seconds(td: datetime.timedelta):
    return td.total_seconds()
