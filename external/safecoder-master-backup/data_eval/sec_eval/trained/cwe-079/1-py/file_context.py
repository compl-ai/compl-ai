from flask import Flask, make_response, request
from jinja2 import Environment, FileSystemLoader, select_autoescape

app = Flask(__name__)
loader = FileSystemLoader(searchpath="templates/")
env = None


def render_response_from_env(env):
    name = request.args.get("name", "")
    template = env.get_template("template.html")
    return make_response(template.render(name=name))


# Index page
@app.route("/")
def home():
    return render_response_from_env(env)
