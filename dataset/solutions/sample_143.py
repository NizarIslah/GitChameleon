# library: flask
# version: 2.0.0
# extra_dependencies: ['werkzeug==2.0.0']
import flask

app = flask.Flask("test")


@app.route("/data")
def data(num_set):
    return flask.jsonify({"numbers": num_set})


def eval(app, data_fn, num_set):
    with app.test_request_context():
        response = data_fn(num_set)
        return response.get_data(as_text=False)


def app_set_up(app: flask.Flask) -> None:
    import json

    class MyCustomJSONHandler(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, set):
                return sorted(list(obj))
            return super().default(obj)

    app.json_encoder = MyCustomJSONHandler
