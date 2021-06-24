import numpy as np
import pandas as pd
import os
import h5py
import cv2
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import cv2 as cv
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.metrics import accuracy_score
from PIL import Image
import pyttsx3

from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import confusion_matrix
import seaborn as sns

from keras.models import load_model
from keras.preprocessing import image
model = load_model('Trained_Model2.h5')
def alarm(msg):
    engine = pyttsx3.init()
    engine.setProperty('rate', 150)
    engine.say(msg)
    engine.runAndWait()

frameWidth = 640  # CAMERA RESOLUTION
frameHeight = 480
brightness = 180
threshold = 80

cap = cv2.VideoCapture(0)
cap.set(3, frameWidth)
cap.set(4, frameHeight)
cap.set(10, brightness)
font = cv2.FONT_HERSHEY_SIMPLEX


def getName(Diff_Pred):
    if(Diff_Pred == 1):
        return "Fire"


while True:
    success, imgOrignal = cap.read()
    # PROCESS IMAGE
    img = np.asarray(imgOrignal)
    img = cv2.resize(img, (160, 160))
    cv2.imshow("Processed Image", img)
    img = img / 255
    img = img.reshape(1, 160, 160, 3)
    cv2.putText(imgOrignal, "CLASS: ", (20, 35), font, 0.75, (0, 0, 255), 2, cv2.LINE_AA)
    cv2.putText(imgOrignal, "PROBABILITY: ", (20, 75), font, 0.75, (0, 0, 255), 2, cv2.LINE_AA)
    # PREDICT IMAGE
    predictions = model.predict(img)
    predictions = int(predictions)
    probabilityValue = np.amax(predictions)

    if(probabilityValue*100 > threshold):
        cv2.putText(imgOrignal, str(predictions) + " "+getName(predictions), (120, 35), font, 0.75, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.putText(imgOrignal, str(round(probabilityValue * 100, 2)) + "%", (180, 75), font, 0.75, (0, 0, 255), 2,cv2.LINE_AA)
        alarm("Fire Alert")
    else:
        cv2.putText(imgOrignal, " No Fire" , (120, 35), font, 0.75, (0, 0, 255), 2,cv2.LINE_AA)
        cv2.putText(imgOrignal, "100%", (180, 75), font, 0.75, (0, 0, 255), 2,cv2.LINE_AA)
    cv2.imshow("Result", imgOrignal)

    if cv2.waitKey(1) and 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
