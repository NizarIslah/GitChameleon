import json
import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import sample_143


import json

app_set_up(app)
app2 = flask.Flask("test2")


@app2.route("/data2")
def data2(num_set):
    return flask.jsonify({"numbers": num_set})


class MyCustomJSONHandler2(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, set):
            return sorted(list(obj))
        return super().default(obj)


app2.json_encoder = MyCustomJSONHandler2
assertion_result = eval(app2, data2, {3, 1, 2, 6, 5, 4}) == eval(
    app, data, {3, 1, 2, 6, 5, 4}
)
assert assertion_result
