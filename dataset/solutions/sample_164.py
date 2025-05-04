# library: flask
# version: 3.0.0
# extra_dependencies: ['numpy==1.25.1']
import flask
import numpy as np

app = flask.Flask("test1")


@app.route("/data")
def data(num_arr):
    return flask.jsonify({"numbers": num_arr})


def eval_app(app, data_fn, num_arr):
    with app.test_request_context():
        response = data_fn(num_arr)
        return response.get_data(as_text=True)


class MyCustomJSONHandler(flask.json.provider.DefaultJSONProvider):
    def default(self, obj: object) -> object:
        if isinstance(obj, np.ndarray):
            unique_vals = np.unique(obj, equal_nan=False)
            return unique_vals.tolist()
        return super().default(obj)


app.json_provider_class = MyCustomJSONHandler
app.json = app.json_provider_class(app)
