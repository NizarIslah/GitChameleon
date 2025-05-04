import json

# Add the parent directory to import sys
import os
import sys

import numpy as np
import pytest
import flask  # <-- Added this import

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from sample_171 import MyCustomJSONHandler, app, data, eval_app
from scipy.stats import hmean


app2 = flask.Flask("test2")


@app2.route("/data2")
def data2(num_arr):
    return flask.jsonify({"numbers": num_arr})


class MyCustomJSONHandler2(flask.json.provider.DefaultJSONProvider):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            res = hmean(obj, axis=1).tolist()
            return res
        return super().default(obj)


app2.json_provider_class = MyCustomJSONHandler2
app2.json = app2.json_provider_class(app2)
assertion_results = eval_app(
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
) == eval_app(
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
