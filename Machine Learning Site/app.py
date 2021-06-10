from flask import *  
import feature_extraction
from tensorflow.keras.models import load_model
import cv2
import numpy as np
import dlib
import pandas as pd
import os, glob
from imutils import face_utils
import argparse
import imutils

app = Flask(__name__)  
 
@app.route('/')  
def upload():  
    return render_template("index.html")  
 
@app.route('/success', methods = ['POST'])  
def success():  
    if request.method == 'POST':  
        f = request.files['file']  
        f.save(f.filename)  
        result = feature_extraction.make_prediction(f.filename)
        return render_template("success.html", name = result)  
  
if __name__ == '__main__':  
    app.run(debug = True)  