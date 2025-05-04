import json
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from sample_164 import MyCustomJSONHandler, app, data, eval_app


app2 = flask.Flask("test2")


@app2.route("/data2")
def data2(num_arr):
    return flask.jsonify({"numbers": num_arr})


class MyCustomJSONHandler2(flask.json.provider.DefaultJSONProvider):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            n_nan = np.sum(np.isnan(obj))
            unique_vals = obj[~np.isnan(obj)]
            unique_vals = np.append(np.unique(unique_vals), [np.nan] * n_nan).tolist()
            return unique_vals
        return super().default(obj)


app2.json_provider_class = MyCustomJSONHandler2
app2.json = app2.json_provider_class(app2)

assertion_results = eval_app(
    app2, data2, np.array([3, 3, 1, np.nan, 2, 6, 5, np.nan])
) == eval_app(app, data, np.array([3, 3, 1, np.nan, 2, 6, 5, np.nan]))
assert assertion_results
