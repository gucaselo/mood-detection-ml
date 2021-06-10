import cv2
import numpy as np
import dlib
import pandas as pd
import os, glob
from imutils import face_utils
import argparse
import imutils

def file_features(path):
    
    # Load the detector
    detector = dlib.get_frontal_face_detector()
    
    # Load the predictor
    predictor = dlib.shape_predictor("static/img_dataset/shape_predictor_68_face_landmarks.dat")

    emotions_features = pd.DataFrame()
    
    # for filepath in glob.glob(path):
        
    # Read the video
    # img = cv2.imread(filepath)
    cap = cv2.VideoCapture(path)
    ret, frame = cap.read()

    # Convert image into grayscale
    gray = cv2.cvtColor(src=frame, code=cv2.COLOR_BGR2GRAY)

    # Use detector to find landmarks
    faces = detector(gray)
    
    for face in faces:
        x1 = face.left() # left point
        y1 = face.top() # top point
        x2 = face.right() # right point
        y2 = face.bottom() # bottom point


        landmarks = predictor(image=gray, box=face)
        
        emptyarray = []
        
        for n in range(0, 68):
            x = landmarks.part(n).x
            y = landmarks.part(n).y
#             print(x)
            # Created a coordinate numpy array
            coordinates = np.array([x, y])
        
            # Append coordinate values (x, y) to emptyarray
            emptyarray.append(coordinates[0])
            emptyarray.append(coordinates[1])

            # Create dataframe
            df = pd.DataFrame({'features':[emptyarray]})
    
        # Append all df from all files/faces
        emotions_features = emotions_features.append(df)
        
    emotions_features['features']= emotions_features['features'].astype(str).str[1:-1]
    emotions_features2 = emotions_features['features'].str.split(",", n = 135, expand = True) 
    emotions_features2.rename(columns= lambda s:f"data{s}", inplace=True )

    emotions_features2 = emotions_features2.apply(pd.to_numeric)
    emotions_features2.fillna(0,inplace=True)
    emotions_features2 = emotions_features2.astype({"data0": int})

    
    emotions_features2.reset_index(drop=True, inplace=True)

    return emotions_features2

def make_prediction(path):
    from tensorflow.keras.models import load_model
    model =load_model("static/img_dataset/image_emotions_model_test.h5")
    
    # Extract image file features
    df = file_features(path)
    
    # Make prediction using imported model
    prediction = model.predict_classes(df)
#     prediction_label = label_encoder.inverse_transform(prediction)
    
    # Emotion labels
    emotion_codes = {
                0: 'Angry',
                1:'Disgust' ,
                2:'Fear',
                3:'Happy',
                4:'Sad',
                5:'Surprise',
                6:'Neutral'
               }
    result = emotion_codes[prediction[0]]
    
    result_dict = {'emotion': result}
    
    return result_dict