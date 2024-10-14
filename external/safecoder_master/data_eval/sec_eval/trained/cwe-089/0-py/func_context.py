# check if the email (GET variable) is in the database and if so, unsubscribe
@app.route("/unsubscribe")
def unsubscribe():
    # get the email (GET variable)
    email = request.args.get("email")
    # get the database connection
    db = MySQLdb.connect(host="localhost", user="root", passwd="root", db="db")
