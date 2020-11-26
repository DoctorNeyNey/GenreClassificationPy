#!/usr/bin/python3.8
import json
import numpy as np
#from pydub import AudioSegment
from sklearn.model_selection import train_test_split
import tensorflow as tf
import librosa
import tensorflow.keras as keras
import matplotlib.pyplot as plt
import math
import soundfile as sf


DATASET_PATH = "data.json"
checkpoint_path = "checkpoints/cp1.ckpt"

answers = ["Metal","Blues","Reggae","Hiphop","Disco","Pop","Jazz","Rock","Classical","Country"]



def load_single(file_path):
    #Audio Processing Values
    SAMPLE_RATE = 22050 #customary value 
    n_mfcc=13
    n_fft=2048
    hop_length=512
    num_segments=10
    print("Loading File")
    try:
        data = []
        #Load Signal
        signal, sr = librosa.load(file_path, sr=SAMPLE_RATE)
        print("File Sucessfully Loaded")

        #Set up Variables
        DURATION = math.floor(librosa.get_duration(filename=file_path))
        SAMPLES_PER_TRACK = SAMPLE_RATE * DURATION
        num_samples_per_segment = 66150

        expected_num_mfcc_vectors_per_segment = math.ceil(num_samples_per_segment / hop_length) 
        num_segments = math.floor(SAMPLES_PER_TRACK /num_samples_per_segment)
        
        #Start Processing Samples
        for s in range(num_segments):
            start_sample = num_samples_per_segment * s 
            finish_sample = start_sample + num_samples_per_segment

            mfcc = librosa.feature.mfcc(signal[start_sample:finish_sample], 
                                    sr=sr,
                                    n_fft=n_fft,
                                    n_mfcc=n_mfcc,
                                    hop_length=hop_length)
            mfcc = mfcc.T

            if len(mfcc) == expected_num_mfcc_vectors_per_segment:
                data.append(mfcc.tolist())
    except:
        print("file {} had an error. Giving Up".format(file_path))
        exit()

    
    print("Successfully Created")
    return data

def predict(model, X, fp):

    #Add new axis to the front since there is only 1
    #Add new axis to the back because we do this at the end need to check and update for reasoning
    scoreSheet = np.zeros(10)
    for i in X:
        temp = i[np.newaxis, ..., np.newaxis]

        prediction = np.argmax(model.predict(temp), axis=1)
        print("{} is {}".format(fp, answers[int(prediction)]))
        scoreSheet[int(prediction)] += 1
    
    print("prediction:")
    print(answers[np.argmax(scoreSheet)])
    print("Final Scores: ")
    for i in range(len(answers)):
        print(answers[i],":", scoreSheet[i])


def build_model(input_shape):

    #create model
    model = keras.Sequential()

    #1st conv layer
    model.add(keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=input_shape))
    model.add(keras.layers.MaxPooling2D((3,3), strides=(2,2), padding='same'))
    model.add(keras.layers.BatchNormalization())
    
    #2nd conv layer
    model.add(keras.layers.Conv2D(32, (3,3), activation='relu'))
    model.add(keras.layers.MaxPooling2D((3,3), strides=(2,2), padding='same'))
    model.add(keras.layers.BatchNormalization())

    #3rd conv layer
    model.add(keras.layers.Conv2D(32, (2,2), activation='relu'))
    model.add(keras.layers.MaxPooling2D((2,2), strides=(2,2), padding='same'))
    model.add(keras.layers.BatchNormalization())

    #flatten the output and feed it into dense layer
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(64, activation='relu'))
    model.add(keras.layers.Dropout(0.3))

    #output layer
    model.add(keras.layers.Dense(10, activation='softmax'))

    return model

if __name__ == "__main__":
    #Load Song File
    print("Enter File Path:")
    fp = input()
    print(fp)
    song_mfcc = load_single(fp)
    
    #Just copy the input shape from cnn_genre_classifier.py
    input_shape = (130, 13, 1)

    #build the CNN net
    model = build_model(input_shape)
    print("Model built")

    #Compile the network 
    optimizer = keras.optimizers.Adam(learning_rate=0.0001)
    model.compile(optimizer=optimizer,
                  loss="sparse_categorical_crossentropy",
                  metrics=['accuracy'])              
    print("Model Compiled")
    
    #Load Previously Stored Weights
    model.load_weights(checkpoint_path)
    print("Saved Weights Loaded")
    song_mfcc = np.array(song_mfcc)
    
    #Predict the value from out song
    print("Testing Song on Network")
    predict(model, song_mfcc, fp)