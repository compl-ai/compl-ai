# function which says hello to the user
@app.route("hello")
def hello():
    username = request.args.get("username")
