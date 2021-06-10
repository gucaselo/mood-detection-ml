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
# from camera import VideoCamera

app = Flask(__name__)  

# camera = cv2.VideoCapture(0)

# def gen_frames():  
#     while True:
#         success, frame = camera.read()  # read the camera frame
        
#         if not success:
#             break
#         else:
#             ret, buffer = cv2.imencode('.jpg', frame)
#             frame = buffer.tobytes()
#             yield (b'--frame\r\n'
#                    b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  # concat frame one by one and show result

@app.route('/')  
def upload():  
    return render_template("index.html") 

# @app.route('/video_feed')
# def video_feed():
#     result = Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')
#     print(result)

#     return result
 
@app.route('/success', methods = ['POST'])  
def success():  
    if request.method == 'POST':  

        if request.files['video'] and request.files['image']:
            videof = request.files['video'] 
            imagef = request.files['image']  
            videof.save(videof.filename)
            imagef.save(imagef.filename) 

            # Run model to predict mood
            # result = feature_extraction.make_prediction(videof.filename)
            result = feature_extraction.video_image_prediction(videof.filename, imagef.filename)
            
            # Remove file after processing
            if os.path.exists(videof.filename and imagef.filename):
                os.remove(videof.filename)
                os.remove(imagef.filename)
            return render_template("success.html", name = result) 

        elif request.files['video']:

            videof = request.files['video']  
            videof.save(videof.filename)

            # Run model to predict mood
            result = feature_extraction.video_prediction(videof.filename)
            
            # Remove file after processing
            if os.path.exists(videof.filename):
                os.remove(videof.filename)
            return render_template("success.html", name = [result]) 

        elif request.files['image']:

            imagef = request.files['image'] 
            imagef.save(imagef.filename) 
            result = feature_extraction.image_prediction(imagef.filename)
            # dicti = {'emotion': 'None'}

            # Remove file after processing
            if os.path.exists(imagef.filename):
                os.remove(imagef.filename)
            # return render_template("success.html", name = dicti)  
            return render_template("success.html", name = [result]) 

        elif request.files['audio']:

            audiof = request.files['audio'] 
            audiof.save(audiof.filename) 
            result = feature_extraction.audio_prediction(audiof.filename)

            # Remove file after processing
            if os.path.exists(audiof.filename):
                os.remove(audiof.filename)
            # return render_template("success.html", name = dicti)  
            return render_template("success.html", name = [result]) 




        
  
if __name__ == '__main__':  
    app.run(debug = True)  