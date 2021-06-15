# Emotion Recognition and Machine Learning

Created three Machine learning models that utilize data to model, fit, and predict emotion across 3 modes: text, voice, and photo/video.

## Description
The models that were created to predict voice, text and image are described below:

* Voice
  * Neural Network
     * Multi-layer Perceptron classifier (MLPClassifier)
  * [Librosa](https://librosa.org/doc/latest/feature.html) to extrat low level features from audio files
  * [RAVDESS Emotional Speech Audio Datset](https://www.kaggle.com/uwrfkaggler/ravdess-emotional-speech-audio)

* Text
  * Multinomial Naive Bayes (MultinomialNB)
  * TfidfVectorizer
  * Natural Language Toolkit (NLTK) (to tokenize and stopwords)
  * [Twitter Text Sentiment Data](http://thinknook.com/twitter-sentiment-analysis-training-corpus-dataset-2012-09-22/)
 
* Image
  * Deep Learning Model.
     * [TensorFlow](https://www.tensorflow.org/install) - Keras
  * Facial Shape Landmark predictor from Italo José [GitHub](https://github.com/italojs/facial-landmarks-recognition/blob/master/shape_predictor_68_face_landmarks.dat)
  * [Python OpenCV](https://docs.opencv.org/master/d6/d00/tutorial_py_root.html)
  * [Dlib](http://dlib.net/python/index.html)
  * [Imutils](https://pypi.org/project/imutils/) (Future implementations)
  * [Image Labeled Dataset FER-2013](https://www.kaggle.com/msambare/fer2013)
 
 

### Libraries and Tools
* [Python OpenCV](https://docs.opencv.org/master/d6/d00/tutorial_py_root.html)
* [Trained Facial Recognition Model](https://github.com/italojs/facial-landmarks-recognition/blob/master/shape_predictor_68_face_landmarks.dat)
* [Dlib](http://dlib.net/python/index.html)
* [Keras](https://www.tensorflow.org/guide/keras/sequential_model)
* [TensorFlow](https://www.tensorflow.org/)
* [NLTK](https://www.nltk.org/install.html)
  * Due to the size of this library further download needs to be done after you pip install it. The first time you run it you need to uncomment the line called # nltk.download()
    Onced you run the cell a pop-up window will appear that will required further download and then you can successfully run the script.
* [Imutils](https://pypi.org/project/imutils/)
* [Librosa](https://librosa.org/doc/latest/feature.html)
* [Numpy](https://numpy.org/install/)
* [Pandas](https://pandas.pydata.org/pandas-docs/stable/getting_started/install.html)
* [Flask](https://flask.palletsprojects.com/en/2.0.x/installation/)


## Datasets
Due to their size these datasets were not included in our repo. Please refer to them using the links below.

* [Image Labeled Dataset FER-2013](https://www.kaggle.com/msambare/fer2013)

* [Twitter Text Sentiment Data](http://thinknook.com/twitter-sentiment-analysis-training-corpus-dataset-2012-09-22/)

* [RAVDESS Emotional Speech Audio Datset](https://www.kaggle.com/uwrfkaggler/ravdess-emotional-speech-audio)
