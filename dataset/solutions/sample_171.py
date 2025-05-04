# library: flask
# version: 3.0.0
# extra_dependencies: ['scipy==1.11.1']
import flask
import numpy as np
from scipy.stats import hmean

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
            res = hmean(obj, axis=1).tolist()
            return res
        return super().default(obj)


app.json_provider_class = MyCustomJSONHandler
app.json = app.json_provider_class(app)
