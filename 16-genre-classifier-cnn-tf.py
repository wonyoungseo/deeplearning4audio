import json
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow.keras as keras
import matplotlib.pyplot as plt

DATA_PATH = "datasets/processed/data_10.json"


import json
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow.keras as keras
import matplotlib.pyplot as plt

DATA_PATH = "datasets/processed/data_10.json"


def load_data(data_path):

    with open(data_path, "r") as fp:
        data = json.load(fp)

    X = np.array(data["mfcc"])
    y = np.array(data["labels"])

    print("Data successfully loaded!")

    return X, y


def prepare_dataset(test_size, validation_size):
    # load data
    X, y = load_data(DATA_PATH)

    # create train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)

    # create train/validation split
    X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train, test_size=validation_size)

    # image data requires 3 dimensional array -> (rows, cols, channels) -> (timebean, MFCCs, 1 channel) -> (130, 13, 1)
    # array[..., new axis]
    X_train = X_train[..., np.newaxis] # 4d array -> (batch_size-num_samples, 130, 13, 1)
    X_validation = X_validation[..., np.newaxis]
    X_test = X_test[..., np.newaxis]

    # return shape -> (nums, height, width, channels)
    return X_train, X_validation, X_test, y_train, y_validation, y_test


def build_model(input_shape):

    # create model
    model = keras.Sequential()

    ## 1st Conv layer
    model.add(keras.layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
    model.add(keras.layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2), padding='same'))
    model.add(keras.layers.BatchNormalization())

    ## 2st Conv layer
    model.add(keras.layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu'))
    model.add(keras.layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2), padding='same'))
    model.add(keras.layers.BatchNormalization())

    ## 3st Conv layer
    model.add(keras.layers.Conv2D(filters=32, kernel_size=(2, 2), activation='relu'))
    model.add(keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same'))
    model.add(keras.layers.BatchNormalization())

    ## flatten the output and feed it into dense layer
    model.add(keras.layers.Flatten())

    ## fully connected layer
    model.add(keras.layers.Dense(units=64, activation='relu'))

    ## dropout
    model.add(keras.layers.Dropout(0.3))

    ## output layer
    model.add(keras.layers.Dense(10, activation='softmax'))

    return model

def predict(model, X, y):

    X = X[np.newaxis, ...]

    prediction = model.predict(X)  # X -> (1, 130, 13, 1)

    predicted_index = np.argmax(prediction, axis=1)
    print("Expected index: {},  Predicted index: {}".format(y, predicted_index))

if __name__ == "__main__":

    # create train, validation and test sets
    X_train, X_val, X_test, y_train, y_val, y_test = prepare_dataset(0.25, 0.2)
    input_shape = (
        X_train.shape[1],  # height
        X_train.shape[2],  # width
        X_train.shape[3]   # channel
    )
    # build CNN
    model = build_model(input_shape)

    # compile the network
    optimizer = keras.optimizers.Adam(learning_rate=0.0001)
    model.compile(optimizer=optimizer,
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])

    # train CNN
    model.fit(X_train, y_train, validation_data=(X_val, y_val), batch_size=32, epochs=30)

    # evaluate CNN on test set
    test_error, test_accuracy = model.evaluate(X_test, y_test, verbose=1)
    print("Accuracy on test set is: {}".format(test_accuracy))

    # make prediction on sample
    X = X_test[100]
    y = y_test[100]

    predict(model, X, y)

