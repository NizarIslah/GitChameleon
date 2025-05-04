# library: flask
# version: 3.0.0
# extra_dependencies: []
import json
import tempfile
from flask import Flask

config_data = {"DEBUG": True, "SECRET_KEY": "secret"}
with tempfile.NamedTemporaryFile(mode="w+", delete=False, suffix=".json") as tmp:
    json.dump(config_data, tmp)
    tmp.flush()
    config_file = tmp.name

app = Flask(__name__)


def load_config(config_file: str) -> None:
    app.config.from_file(config_file, load=json.load)
