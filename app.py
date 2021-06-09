# 1. import Flask and dependencies
from flask import Flask, jsonify, render_template
from sqlalchemy.ext.automap import automap_base
from sqlalchemy.orm import Session
from sqlalchemy import create_engine, func
import numpy as np
import sqlite3
from pathlib import Path
import pandas as pd
import json

####Importing/Defining model data


#################################################
# Flask Setup
#################################################
# Create an app
app = Flask(__name__)

#################################################
# Route to homepage: index.html
#################################################
# Define what to do when a user hits the index route
@app.route("/")
def welcome():

    return render_template("index.html")