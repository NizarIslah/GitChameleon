import json

# Add the parent directory to import sys
import os
import sys
import unittest

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sample_144 import app, app_set_up, data, eval


import json

app_set_up(app)
app2 = flask.Flask("test2")


@app2.route("/data2")
def data2(num_set):
    return flask.jsonify({"numbers": num_set})


class MyCustomJSONHandler2(flask.json.provider.DefaultJSONProvider):
    def default(self, obj):
        if isinstance(obj, set):
            return sorted(list(obj))
        return super().default(obj)


app2.json_provider_class = MyCustomJSONHandler2
app2.json = app2.json_provider_class(app2)

assertion_result = eval(app2, data2, {3, 1, 2, 6, 5, 4}) == eval(
    app, data, {3, 1, 2, 6, 5, 4}
)
assert assertion_result
