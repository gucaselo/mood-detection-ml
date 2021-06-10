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

        if request.files['video'] and request.files['image']:
            videof = request.files['video'] 
            imagef = request.files['image']  
            videof.save(videof.filename)
            imagef.save(imagef.filename) 

            # Run model to predict mood
            result = feature_extraction.make_prediction(videof.filename)
            
            # Remove file after processing
            if os.path.exists(videof.filename and imagef.filename):
                os.remove(videof.filename)
                os.remove(imagef.filename)
            return render_template("success.html", name = result) 

        elif request.files['video']:

            videof = request.files['video']  
            videof.save(videof.filename)

            # Run model to predict mood
            result = feature_extraction.make_prediction(videof.filename)
            
            # Remove file after processing
            if os.path.exists(videof.filename):
                os.remove(videof.filename)
            return render_template("success.html", name = result) 

        elif request.files['image']:

            imagef = request.files['image'] 
            imagef.save(imagef.filename) 
            dicti = {'emotion': 'None'}

            # Remove file after processing
            if os.path.exists(imagef.filename):
                os.remove(imagef.filename)
            return render_template("success.html", name = dicti)  




        
  
if __name__ == '__main__':  
    app.run(debug = True)  