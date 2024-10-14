#!/bin/python3

import os
import subprocess
import sys

import MySQLdb
from flask import Flask, redirect, request

app = Flask(__name__)


def get_db_connection():
    db = MySQLdb.connect(host="localhost", user="root", passwd="root", db="db")
    return db


@app.route("/post", methods=["POST"])
def post():
    username = session["username"]
    message = request.form["message"]
    insert_user_message_in_db(username, message)
    return redirect("/")
