import json

# Add the parent directory to import sys
import os
import sys
import unittest

import numpy as np
from scipy.stats import hmean
import flask  # <-- Added import

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sample_172 import MyCustomJSONHandler, app, data, eval


app2 = flask.Flask("test2")


@app2.route("/data2")
def data2(num_arr):
    return flask.jsonify({"numbers": num_arr})


class MyCustomJSONHandler2(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            res = np.zeros((obj.shape[0], 1))
            for i_arr in range(obj.shape[0]):
                if np.isnan(obj[i_arr]).any():
                    res[i_arr] = np.nan
                else:
                    res[i_arr] = hmean(obj[i_arr])
            res = res.flatten().tolist()
            return res
        return super().default(obj)


app2.json_encoder = MyCustomJSONHandler2
assertion_results = eval(
    app2,
    data2,
    np.array(
        [
            [
                3,
                3,
                np.nan,
            ],
            [np.nan, 2, 4],
            [1, 2, 1],
        ]
    ),
) == eval(
    app,
    data,
    np.array(
        [
            [
                3,
                3,
                np.nan,
            ],
            [np.nan, 2, 4],
            [1, 2, 1],
        ]
    ),
)
assert assertion_results
