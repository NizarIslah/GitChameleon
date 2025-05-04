import json

# Add the parent directory to import sys
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from sample_169 import MyCustomJSONHandler, app, data, eval_app


app2 = flask.Flask("test2")


@app2.route("/data2")
def data2(num_arr):
    return flask.jsonify({"numbers": num_arr})


class MyCustomJSONHandler2(flask.json.provider.DefaultJSONProvider):
    def default(self, obj):
        if (
            isinstance(obj, np.ndarray)
            and len(obj.shape) == 3
            and obj.shape[-1] == obj.shape[-2]
        ):
            res = np.zeros(obj.shape[0])
            for i in range(obj.shape[0]):
                res[i] = linalg.det(obj[i])
            return res.tolist()
        return super().default(obj)


app2.json_provider_class = MyCustomJSONHandler2
app2.json = app2.json_provider_class(app2)
a = np.random.random((6, 3, 3))

assertion_results = eval_app(app2, data2, a) == eval_app(app, data, a)
assert assertion_results
