# library: flask
# version: 2.0.1
# extra_dependencies: ['werkzeug==2.0.0']
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
    app.config.from_json(config_file)
