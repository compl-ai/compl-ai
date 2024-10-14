import os
import subprocess
import sys

import MySQLdb
from flask import Flask, redirect, request

app = Flask(__name__)
