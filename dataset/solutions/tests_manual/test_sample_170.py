import json
import os
import sys
import unittest

import flask
import numpy as np
from scipy import linalg

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from sample_170 import MyCustomJSONHandler, app, data, eval


app2 = flask.Flask("test2")


@app2.route("/data2")
def data2(num_arr):
    return flask.jsonify({"numbers": num_arr})


class MyCustomJSONHandler2(json.JSONEncoder):
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


app2.json_encoder = MyCustomJSONHandler2
a = np.random.random((6, 3, 3))

assertion_results = eval(app2, data2, a) == eval(app, data, a)
assert assertion_results
