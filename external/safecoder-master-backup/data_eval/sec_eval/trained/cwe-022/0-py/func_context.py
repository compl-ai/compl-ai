@app.route("/read")
def read():
    requested_file_name = request.args.get("filename")
    safe_dir = "/safe/"

    # read the requested file from the safe directory
