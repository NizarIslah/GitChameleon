# library: flask
# version: 3.0.0
# extra_dependencies: ['scipy==1.11.1']
import flask
import numpy as np
from scipy import linalg

app = flask.Flask("test1")


@app.route("/data")
def data(num_list):
    return flask.jsonify({"numbers": num_list})


def eval_app(app, data_fn, num_arr):
    with app.test_request_context():
        response = data_fn(num_arr)
        return response.get_data(as_text=True)


class MyCustomJSONHandler(flask.json.provider.DefaultJSONProvider):
    def default(self, obj):
        if (
            isinstance(obj, np.ndarray)
            and len(obj.shape) == 3
            and obj.shape[-1] == obj.shape[-2]
        ):
            res = linalg.det(obj)
            return res.tolist()
        return super().default(obj)


app.json_provider_class = MyCustomJSONHandler
app.json = app.json_provider_class(app)
