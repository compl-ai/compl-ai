import flask
from flask import Flask, request
from markupsafe import escape

app = Flask(__name__)


@app.route("/")
def index():
    return "<p>Hello, World!</p>"
