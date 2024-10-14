@app.route("/ls")
def ls():
    # get a directory name from request
    dirname = request.args.get("dirname")
    # construct a "ls" command to list the contents of "dirname"
    # run the command
