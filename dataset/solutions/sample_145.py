# library: flask
# version: 2.0.0
# extra_dependencies: ['werkzeug==2.0.0']
from flask import Flask, send_file
from io import BytesIO

app1 = Flask(__name__)


def get_content_disp(app, download_fn):
    with app.test_request_context():
        response = download_fn()
    content_disp = response.headers.get("Content-Disposition")
    return content_disp


@app1.route("/download")
def download():
    data = BytesIO(b"Hello, World!")
    attachment_filename = "hello.txt"
    return send_file(data, as_attachment=True, attachment_filename=attachment_filename)
