import json

# Add the parent directory to import sys
import os
import sys
import unittest

import numpy as np

# from flask import Flask # Flask class not directly instantiated here, app object is imported

# The original sys.path.append logic is assumed to be correct for the project structure
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sample_163 import MyCustomJSONHandler


app2 = flask.Flask("test2")


@app2.route("/data2")
def data2(num_arr):
    return flask.jsonify({"numbers": num_arr})


class MyCustomJSONHandler2(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            n_nan = np.sum(np.isnan(obj))
            unique_vals = obj[~np.isnan(obj)]
            unique_vals = np.append(np.unique(unique_vals), [np.nan] * n_nan).tolist()
            return unique_vals
        return super().default(obj)


app2.json_encoder = MyCustomJSONHandler2
assertion_results = eval(
    app2, data2, np.array([3, 3, 1, np.nan, 2, 6, 5, np.nan])
) == eval(app, data, np.array([3, 3, 1, np.nan, 2, 6, 5, np.nan]))
assert assertion_results
