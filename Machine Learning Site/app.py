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
def index():  
    return render_template("index.html") 
 
@app.route('/text')  
def text():  
    return render_template("text.html") 

@app.route('/audio')  
def audio():  
    return render_template("record.html") 
 
@app.route('/image')   
def image():  
    return render_template("image.html")
 
@app.route('/video')  
def video():  
    return render_template("video.html") 
 
@app.route('/about') 
def about():  
    return render_template("aboutus.html") 

@app.route('/feedback') 
def feedback():  
    return render_template("leave_feedback.html") 


@app.route('/emotion', methods = ['POST'])   
def emotion():  
    if request.method == 'POST':  
        if request.form['form-name'] == "image":
            if request.files['image']:
                imagef = request.files['image'] 
                imagef.save(imagef.filename) 
                result = feature_extraction.image_prediction(imagef.filename)

                # Remove file after processing
                if os.path.exists(imagef.filename):
                    os.remove(imagef.filename)

                return render_template("image.html", name = result)
            else:
                return render_template("image.html")

        elif request.form['form-name'] == "text":
            if request.form['text']:
                text = request.form['text']  
                result = feature_extraction.text_cleaning(text)

                return render_template("text.html", name = result) 
            else:
                return render_template("text.html") 

        elif request.form['form-name'] == "audio":
            if request.files['audio']:
                audiof = request.files['audio'] 
                audiof.save(audiof.filename) 
                result = feature_extraction.audio_prediction(audiof.filename)

                # Remove file after processing
                if os.path.exists(audiof.filename):
                    os.remove(audiof.filename)

                return render_template("record.html", name = result) 
            else:
                return render_template("record.html") 

        elif request.form['form-name'] == "video":
            if request.files['video']:
                videof = request.files['video'] 
                videof.save(videof.filename) 
                result = feature_extraction.video_prediction(videof.filename)

                # Remove file after processing
                if os.path.exists(videof.filename):
                    os.remove(videof.filename)

                return render_template("video.html", name = result) 
            else:
                return render_template("video.html") 

        elif request.form['form-name'] == "feedback":
            if request.form['feedback']:
                text = request.form['feedback']  
                response = feature_extraction.feedback(text)

                return render_template("leave_feedback.html", name = response) 
            else:
                return render_template("leave_feedback.html") 

# @app.route('/emotionimage', methods = ['POST'])   
# def emotionimage():  
#     if request.method == 'POST':  
#         if request.files['image']:
#         # if request.form['form-name'] == "image":
#             imagef = request.files['image'] 
#             imagef.save(imagef.filename) 
#             result = feature_extraction.image_prediction(imagef.filename)

#             # Remove file after processing
#             if os.path.exists(imagef.filename):
#                 os.remove(imagef.filename)

#             return render_template("image.html", name = result)
#         else:
#             return render_template("image.html")

# @app.route('/emotiontext', methods = ['POST'])   
# def emotiontext():  
#     if request.method == 'POST':
#         if request.form['text']:
#             text = request.form['text']  
#             result = feature_extraction.text_cleaning(text)

#             return render_template("text.html", name = result) 
#         else:
#             return render_template("text.html") 

# @app.route('/emotionaudio', methods = ['POST'])   
# def emotionaudio():  
#     if request.method == 'POST':
#         if request.files['audio']:
#             audiof = request.files['audio'] 
#             audiof.save(audiof.filename) 
#             result = feature_extraction.audio_prediction(audiof.filename)

#             # Remove file after processing
#             if os.path.exists(audiof.filename):
#                 os.remove(audiof.filename)

#             return render_template("record.html", name = result) 
#         else:
#             return render_template("record.html") 

# @app.route('/emotionvideo', methods = ['POST'])   
# def emotionvideo():  
#     if request.method == 'POST':
#         if request.files['video']:
#             videof = request.files['video'] 
#             videof.save(videof.filename) 
#             result = feature_extraction.video_prediction(videof.filename)

#             # Remove file after processing
#             if os.path.exists(videof.filename):
#                 os.remove(videof.filename)

#             return render_template("video.html", name = result) 
#         else:
#             return render_template("video.html") 

# @app.route('/leftfeedback', methods = ['POST'])   
# def leftfeedback():  
#     if request.method == 'POST':
#         if request.form['feedback']:
#             text = request.form['feedback']  
#             response = feature_extraction.feedback(text)

#             return render_template("leave_feedback.html", name = response) 
#         else:
#             return render_template("leave_feedback.html") 


# @app.route('/data', methods = ['POST'])
# def data():
#     if request.method == 'POST':  
#         # emotion = []
#         if request.files['image']:
#             imagef = request.files['image'] 
#             imagef.save(imagef.filename) 
#             result = feature_extraction.image_prediction(imagef.filename)
#             # result_dict = {'emotion': result}
#             # emotion.append(result_dict)

#             # Remove file after processing
#             # if os.path.exists(imagef.filename):
#             #     os.remove(imagef.filename)
            
#             # return jsonify(emotion)
#             return render_template("image2.html", name = result)   
 
# @app.route('/success', methods = ['POST'])  
# def success():  
#     if request.method == 'POST':  

        # if request.files['video'] and request.files['image']:
        #     videof = request.files['video'] 
        #     imagef = request.files['image']  
        #     videof.save(videof.filename)
        #     imagef.save(imagef.filename) 

        #     # Run model to predict mood
        #     # result = feature_extraction.make_prediction(videof.filename)
        #     result = feature_extraction.video_image_prediction(videof.filename, imagef.filename)
            
        #     # Remove file after processing
        #     if os.path.exists(videof.filename and imagef.filename):
        #         os.remove(videof.filename)
        #         os.remove(imagef.filename)
        #     return render_template("success.html", name = result) 

        # elif request.files['video']:

        #     videof = request.files['video']  
        #     videof.save(videof.filename)

        #     # Run model to predict mood
        #     result = feature_extraction.video_prediction(videof.filename)
            
        #     # Remove file after processing
        #     if os.path.exists(videof.filename):
        #         os.remove(videof.filename)
        #     return render_template("success.html", name = [result]) 

        # if request.files['image']:

        #     imagef = request.files['image'] 
        #     imagef.save(imagef.filename) 
        #     result = feature_extraction.image_prediction(imagef.filename)
        #     # dicti = {'emotion': 'None'}

        #     # Remove file after processing
        #     if os.path.exists(imagef.filename):
        #         os.remove(imagef.filename)
        #     # return render_template("success.html", name = dicti)  
        #     return render_template("success.html", name = [result]) 

        # elif request.files['audio']:

        #     audiof = request.files['audio'] 
        #     audiof.save(audiof.filename) 
        #     result = feature_extraction.audio_prediction(audiof.filename)

        #     # Remove file after processing
        #     if os.path.exists(audiof.filename):
        #         os.remove(audiof.filename)
        #     # return render_template("success.html", name = dicti)  
        #     return render_template("success.html", name = [result]) 


        # elif request.form['text']:

        #     text = request.form['text']  
        #     result = feature_extraction.text_cleaning(text)

        #     return render_template("success.html", name = [result])   


  
if __name__ == '__main__':  
    app.run(debug = True)  