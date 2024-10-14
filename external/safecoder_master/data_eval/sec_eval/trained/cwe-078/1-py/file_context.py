import os
import subprocess
import sys

from flask import Flask, redirect, request

app = Flask(__name__)
