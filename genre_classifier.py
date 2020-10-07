import json
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow.keras as keras

DATASET_PATH = "data.json"


def load_data(dataset_path):
    with open(dataset_path, "r") as fp:
        data = json.load(fp)
    
    #convert lists into numpy arrays
    inputs = np.array(data["mfcc"])
    targets = np.array(data["labels"])

    return inputs, targets

# Steps:
# 1.Load our data
# 2.Create a train data set and test data set
# 3.Build neural network
# 4.Compile Network
# 5.Train netwrok

if __name__ == "__main__":
    #load data
    inputs, targets = load_data(DATASET_PATH)

    inputs_train, inputs_test, targets_train, targets_test = train_test_split(inputs, targets, test_size = 0.3)

    model = keras.Sequential([
        #input layer
        keras.layers.Flatten(input_shape=(inputs.shape[1], inputs.shape[2])),

        # 1st hidden layer
        #ReLU: Rectified Linear Unit. function: if h < 0 then value is 0 if h >= 0 then the value is h. (no negative numbers allowed)
        #Why use ReLU: 0.Good for training 1. Better convergence 2. Reduced likelihood of vanishing gradient 3
        #Why Not Sigmiod: Sigmiod has it's values "vanish" since it's numbers are very small so multiplying them will cause them to shrink. Is not a problem in smaller
        #nerual networks but in larger complex networks it loses tiny but important information. LeRU doesn't suffer from this problem.
        
        # 1st hidden layer
        keras.layers.Dense(512, activation="relu"),

        # 2nd
        keras.layers.Dense(256, activation="relu"),

        # 3rd
        keras.layers.Dense(64, activation="relu"),

        #as you can see our layers have a ton of connections in each of our layers

        # output layer
        # 10 neurons 1 for each genre
        # 
        keras.layers.Dense(10, activation="softmax")
    ])
    #compile
    optimizer = keras.optimizers.Adam(learning_rate=0.0001)
    model.compile(optimizer=optimizer, 
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    model.summary()

    # train network

    model.fit(inputs_train, targets_train,
              validation_data=(inputs_test, targets_test),
              #number of times we train
              epochs=50,
              #what is batchsize: how much training we do before updating weights
              #Stochastic: batchsize = 1 quick but inaccurate because one wrong werid data point can mess up weights
              #Full Batch: batchsize = n/input size super slow when on huge datasets, but is very accurate 
              #Mini-batch: compute on a subset, best of two worlds much faster than full batch and much more accurrate than stochastic
              batch_size=32)
 