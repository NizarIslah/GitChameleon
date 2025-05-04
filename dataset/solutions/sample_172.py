# library: flask
# version: 2.0.0
# extra_dependencies: ['scipy==1.8.1', 'Werkzeug==2.0.0']
import flask
import json
import numpy as np
from scipy.stats import hmean

app = flask.Flask("test1")


@app.route("/data")
def data(num_arr):
    return flask.jsonify({"numbers": num_arr})


def eval(app, data_fn, num_arr):
    with app.test_request_context():
        response = data_fn(num_arr)
        return response.get_data(as_text=False)


class MyCustomJSONHandler(json.JSONEncoder):
    def default(self, obj: object) -> object:
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


app.json_encoder = MyCustomJSONHandler
