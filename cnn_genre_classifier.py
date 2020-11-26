#!/usr/bin/python3.8
import json
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
import tensorflow.keras as keras
import matplotlib.pyplot as plt


DATASET_PATH = "data.json"
answers = ["metal","blues","reggae","hiphop","disco","pop","jazz","rock","classical","country"]


def plot_history(history):
    """Plots accuracy/loss for training/validation set as a function of the epochs
        :param history: Training history of model
        :return:
    """

    fig, axs = plt.subplots(2)

    # create accuracy sublpot
    axs[0].plot(history.history["accuracy"], label="train accuracy")
    axs[0].plot(history.history["val_accuracy"], label="test accuracy")
    axs[0].set_ylabel("Accuracy")
    axs[0].legend(loc="lower right")
    axs[0].set_title("Accuracy eval")

    # create error sublpot
    axs[1].plot(history.history["loss"], label="train error")
    axs[1].plot(history.history["val_loss"], label="test error")
    axs[1].set_ylabel("Error")
    axs[1].set_xlabel("Epoch")
    axs[1].legend(loc="upper right")
    axs[1].set_title("Error eval")

    plt.show()

def load_data(dataset_path):
    with open(dataset_path, "r") as fp:
        data = json.load(fp)
    
    #convert lists into numpy arrays
    inputs = np.array(data["mfcc"])
    targets = np.array(data["labels"])

    return inputs, targets

def prepare_datasets(test_size, validation_size):

    #load data
    X, y = load_data(DATASET_PATH)

    #create  train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)

    #create train/validation split
    X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train, test_size = validation_size)

    # 3d array for tensor flow for each sample, but we have 2d array
    X_train = X_train[..., np.newaxis] #now a 4d array {num of samples, 130, 13. 1}
    X_validation = X_validation[..., np.newaxis]
    X_test = X_test[..., np.newaxis]
    print(X_test.shape)

    return X_train, X_validation, X_test, y_train, y_validation, y_test

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

def predict(model, X, y):

    X = X[np.newaxis, ...] 
    prediction = model.predict(X)
    predicted_index = np.argmax(prediction, axis=1)
    print("Target: {}, Predicted label: {}".format(answers[int(y)], answers[int(predicted_index)]) )

if __name__ == "__main__":
    #Create Train, validation and test sets
    X_train, X_validation, X_test, y_train, y_validation, y_test = prepare_datasets(0.25, 0.2)
    #print(X_train)
    
    #build the CNN net
    input_shape = (X_train.shape[1], X_train.shape[2], 1)

    model = build_model(input_shape)

    #compile the network
    optimizer = keras.optimizers.Adam(learning_rate=0.0001)
    model.compile(optimizer=optimizer,
                  loss="sparse_categorical_crossentropy",
                  metrics=['accuracy'])

    #Create Checkpoint for saving
    checkpoint_path = "checkpoints/cp1.ckpt"
    #checkpoint_dir = os.path.dirname(checkpoint_path)
    #cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
    #                                            save_weights_only=True,
     #                                           verbose=1)

    # train the CNN
    #model.fit(X_train, y_train, validation_data=(X_validation, y_validation), batch_size=32, epochs=30)
    #print(model, X_test[0], y_test[0])
    #print("erm file saved?")
    #evaluate the CNN on the test set
    checkpoint_path = "checkpoints/cp1.ckpt"
    model.load_weights(checkpoint_path)

    test_error, test_accuracy = model.evaluate(X_test, y_test, verbose=1)
    print("Accuracy on test set is: {}".format(test_accuracy))
    predict(model, X_test[0], y_test[0])
