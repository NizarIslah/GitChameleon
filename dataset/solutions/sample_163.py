# library: flask
# version: 2.0.0
# extra_dependencies: ['numpy==1.21.6', 'werkzeug==2.0.0']
import flask
import json
import numpy as np

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
            n_nan = np.sum(np.isnan(obj))
            unique_vals = obj[~np.isnan(obj)]
            unique_vals = np.append(np.unique(unique_vals), [np.nan] * n_nan).tolist()
            return unique_vals
        return super().default(obj)


app.json_encoder = MyCustomJSONHandler
