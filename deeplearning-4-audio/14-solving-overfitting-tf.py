import json
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow.keras as keras
import matplotlib.pyplot as plt

DATA_PATH = "../datasets/processed/data_10.json"


def load_data(data_path):

    with open(data_path, "r") as fp:
        data = json.load(fp)

    X = np.array(data["mfcc"])
    y = np.array(data["labels"])

    print("Data successfully loaded!")

    return X, y


def plot_history(history):

    fig, axs = plt.subplots(2)

    axs[0].plot(history.history["accuracy"], label="train accuracy")
    axs[0].plot(history.history["val_accuracy"], label="test accuracy")
    axs[0].set_ylabel("Accuracy")
    axs[0].set_xlabel("Epoch")
    axs[0].legend(loc="lower right")
    axs[0].set_title("Accuracy eval")

    axs[1].plot(history.history["loss"], label="train error")
    axs[1].plot(history.history["val_loss"], label="test error")
    axs[1].set_ylabel("Error")
    axs[1].set_xlabel("Epoch")
    axs[1].legend(loc="upper right")
    axs[1].set_title("Error eval")

    plt.show()

if __name__ == "__main__":

    X, y = load_data(DATA_PATH)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

    model = keras.Sequential(
        [
            keras.layers.Flatten(input_shape=(X.shape[1], X.shape[2])),

            keras.layers.Dense(512, activation="relu", kernel_regularizer=keras.regularizers.l2(0.001)),
            keras.layers.Dropout(0.3),

            keras.layers.Dense(256, activation="relu", kernel_regularizer=keras.regularizers.l2(0.001)),
            keras.layers.Dropout(0.3),

            keras.layers.Dense(64, activation="relu", kernel_regularizer=keras.regularizers.l2(0.001)),
            keras.layers.Dropout(0.3),

            keras.layers.Dense(10, activation="softmax"),
        ]
    )

    optimizer = keras.optimizers.Adam(learning_rate=0.0001)
    model.compile(
        optimizer=optimizer,
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.summary()

    history = model.fit(
        X_train, y_train, validation_data=(X_test, y_test), batch_size=32, epochs=100
    )

    plot_history(history)



## Notes on preventing overfitting
# 1. Simple architecture - remove layers, decrease # of neurons
# 2. Data augmentation - (in case of audios) pitch shifting, time stretching, adding noise
# 3. Early stopping - choose rules to stop training at certain epoch or certain accuracy & loss
# 4. (*)Dropout - randomly drop neurons while training and increase model robustness (dropout ratio 0.1~0.5)
# 5. (*)Regularization - add penalty to error function. punish large weights. L1 & L2