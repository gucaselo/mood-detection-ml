# 1. import Flask and dependencies
from flask import Flask, jsonify, render_template
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

# #################################################
# # Team Index
# #################################################
# @app.route("/voice")
# def voice():

#     return render_template("voice.html")

# @app.route("/face")
# def face():

#     return render_template("face.html")

# #################################################
# # Route to obtain voice recording data
# #################################################
# @app.route("/getvoice")
# def getVoice():
#     #################################################
#     # Getting data into json object
#     #################################################
#     res = engine.execute("SELECT lat, lng from firemap")
#     data = json.dumps([dict(r) for r in res])

#     return data
# #################################################
# # Route to obtain face recording data
# #################################################
# @app.route("/getface")
# def getFace():
#     #################################################
#     # Getting data into json object
#     #################################################
#     res = engine.execute("SELECT lat, lng from firemap")
#     data = json.dumps([dict(r) for r in res])

#     return data


#################################################
#End of App
#################################################    
if __name__ == "__main__":
    app.run(debug=True)