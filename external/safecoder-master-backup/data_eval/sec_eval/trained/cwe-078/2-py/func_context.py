# ping a URL to see if it is alive using "ping" system command
@app.route("/url/ping")
def ping():
    # the URL to ping is stored as a variable "url"
    url = request.args.get("url")
    # run the ping command using the system command "ping"
