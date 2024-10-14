# return the requested image from the folder "/images"
@app.route("/img")
def img():
    img_name = request.args.get("img_name")
