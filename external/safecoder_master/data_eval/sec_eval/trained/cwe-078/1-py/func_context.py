@app.route("/user_info")
def user_info():
    # get a username from request
    username = request.args.get("username")
    # construct a command to get the userinfo of the requested username
    # then execute the command
