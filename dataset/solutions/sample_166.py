# library: flask
# version: 3.0.0
# extra_dependencies: ['numpy==1.25.1']
import flask
import numpy as np
import warnings
from numpy import fastCopyAndTranspose

warnings.filterwarnings("error")

app = flask.Flask("test1")


@app.route("/data")
def data(num_list):
    return flask.jsonify({"numbers": num_list})


def eval_app(app, data_fn, num_arr):
    with app.test_request_context():
        response = data_fn(num_arr)
        return response.get_data(as_text=True)


class MyCustomJSONHandler(flask.json.provider.DefaultJSONProvider):
    def default(self, obj: object) -> object:
        if isinstance(obj, np.ndarray):
            res = obj.T.copy().flatten().tolist()
            return res
        return super().default(obj)


app.json_provider_class = MyCustomJSONHandler
app.json = app.json_provider_class(app)
