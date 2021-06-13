import cv2
import numpy as np
import dlib
import pandas as pd
import os, glob
from imutils import face_utils
import argparse
import imutils

#--------------------------------------------------------#
#                       Video Mood                       #
#--------------------------------------------------------#
def extract_features_video(path):
    
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

def video_prediction(path):
    from tensorflow.keras.models import load_model
    model =load_model("static/img_dataset/image_emotions_model_test.h5")
    
    # Extract image file features
    df = extract_features_video(path)
    
    # Make prediction using imported model
    prediction = model.predict_classes(df)
#     prediction_label = label_encoder.inverse_transform(prediction)
    
    # Emotion labels
    emotion_codes = {
                0:'Angry',
                1:'Disgust' ,
                2:'Fear',
                3:'Happy',
                4:'Sad',
                5:'Surprise',
                6:'Neutral'
               }
    result = emotion_codes[prediction[0]]
    
    # result_dict = {'video': result}
    result_dict = {'video': result}
    
    return result_dict

#--------------------------------------------------------#
#                       Image Mood                       #
#--------------------------------------------------------#

def extract_features_image(path):
    
    # Load the detector
    detector = dlib.get_frontal_face_detector()
    
    # Load the predictor
    predictor = dlib.shape_predictor("static/img_dataset/shape_predictor_68_face_landmarks.dat")

    emotions_features = pd.DataFrame()
    
    # for filepath in glob.glob(path):
        
    # Read the video
    img = cv2.imread(path)
    # cap = cv2.VideoCapture(path)
    # ret, frame = cap.read()

    # Convert image into grayscale
    # gray = cv2.cvtColor(src=frame, code=cv2.COLOR_BGR2GRAY)
    gray = cv2.cvtColor(src=img, code=cv2.COLOR_BGR2GRAY)

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

def image_prediction(path):
    from tensorflow.keras.models import load_model
    model =load_model("static/img_dataset/image_emotions_model_test.h5")
    
    # Extract image file features
    df = extract_features_image(path)
    
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

    # emotion_codes = {
    #         0: ['Angry', 'red'],
    #         1:['Disgust', 'greenyellow'],
    #         2:['Fear', 'orange'],
    #         3:['Happy', 'darkorchid'],
    #         4:['Sad', 'yellow'], 
    #         5:['Surprise', 'blue'],
    #         6:['Neutral', 'gray']
    #        }

    result = emotion_codes[prediction[0]]
    # color = emotion_codes[prediction[0][1]]
    
    result_dict = {'image': result}
    
    return result_dict


#--------------------------------------------------------#
#                       Audio Mood                       #
#--------------------------------------------------------#

# Extract features (mfcc, chroma, mel) from a sound file
def extract_feature(file_name, mfcc, chroma, mel):
    import librosa
    import soundfile
    import os, glob, pickle
    import numpy as np
    from sklearn.model_selection import train_test_split
    from sklearn.neural_network import MLPClassifier
    from sklearn.metrics import accuracy_score
    import pandas as pd

    with soundfile.SoundFile(file_name) as sound_file:
        X = sound_file.read(dtype="float32")
        sample_rate=sound_file.samplerate
        if chroma:
            stft=np.abs(librosa.stft(X))
        result=np.array([])
        if mfcc:
            mfccs=np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T, axis=0)
            result=np.hstack((result, mfccs))
        if chroma:
            chroma=np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T,axis=0)
            result=np.hstack((result, chroma))
        if mel:
            mel=np.mean(librosa.feature.melspectrogram(X, sr=sample_rate).T,axis=0)
            result=np.hstack((result, mel))
    return result

def audio_prediction(path):
    import os, glob, pickle
    feature = extract_feature(path, mfcc=True, chroma=True, mel=True)
    filename = 'static/voice_dataset/voice_emotions_model.sav'
    model = pickle.load(open(filename, 'rb'))
    
    result = model.predict(feature.reshape(1, -1))
    
    result_dict = {'audio':result[0]}
    
    return result_dict

# To be used as a reference only
# emotions={
#       '01':'neutral',
#       '02':'neutral',
#       '03':'happy',
#       '04':'sad',
#       '05':'angry',
#       '06':'fearful',
#       '07':'disgust',
#       '08':'surprised'
# }




#--------------------------------------------------------#
#                   Video & Image Mood                   #
#--------------------------------------------------------#

def video_image_prediction(video, image):
    from collections import defaultdict
    # Video prediction
    video = video_prediction(video)

    # Image prediction
    image = image_prediction(image)

    result = [video, image]
    # result_dict = {'image': result}
    
    return result


#--------------------------------------------------------#
#                     Text Sentiment                     #
#--------------------------------------------------------#

def text_cleaning(data):
    import pandas as pd
    import numpy as np
    import nltk
    # nltk.download()
    from nltk import word_tokenize
    from nltk.corpus import stopwords
    import string
    import pickle
    
    # load the model from disk
    model_filename = 'static/text_dataset/text_emotions_model.sav'
    model = pickle.load(open(model_filename, 'rb'))

    # load the vectorizer from disk
    vectorizer_filename = 'static/text_dataset/tfidf_vect.pk'
    tfidf_vect = pickle.load(open(vectorizer_filename, 'rb'))

    text = data.lower()
    cleaned = [char for char in text if char not in string.punctuation]
    cleaned = "".join(cleaned)
    result = np.array([cleaned])
    
    result_prediction = text_features(result, tfidf_vect, model)
    
    emotion = {'text':result_prediction}
    
    return emotion


def text_features(text, tfidf_vect, model):
    import pandas as pd
    import numpy as np
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.model_selection import train_test_split
    from sklearn.naive_bayes import MultinomialNB
    
    text_vect = tfidf_vect.transform(text).toarray()
    
    emotion = model.predict(text_vect.reshape(1, -1))[0]
    
    emotions = {0:'Negative',
                1:'Positive'}
    
    result = emotions[emotion]
    
    return result
