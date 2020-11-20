import json
import numpy as np
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from torchsummary import summary

DATA_PATH = "datasets/processed/data_10.json"


def load_data(data_path):

    with open(data_path, "r") as fp:
        data = json.load(fp)

    X = np.array(data["mfcc"])
    y = np.array(data["labels"])

    return X, y


def reshape_data(data):
    """
    input:
        shape -> (num_samples, height, width)

    output:
        shape -> (num_samples, channels, height, width)
    """

    data = data[..., np.newaxis]  # -> (nums, height, width, channels)
    shape = data.shape
    data = data.reshape(shape[0], shape[3], shape[1], shape[2])  # (nums, channels, height, width)

    return data


def prepare_dataset(test_size, validation_size):
    # load data
    X, y = load_data(DATA_PATH)

    # create train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)

    # create train/validation split
    X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train, test_size=validation_size)
    # returns row (nums, rows, cols)

    # image data requires 3 dimensional array -> (nums, height, width, channels) -> (#samples, timebean, MFCCs, 1 channel) -> (#, 130, 13, 1)
    # Keras CNN input_shape -> (nums, height, width, channels)
    # Pytorch CNN input_shape -> uses NCHW (nums, channel, height, width)
    # create channel dimension: array[..., new axis]
    X_train = reshape_data(X_train)
    X_validation = reshape_data(X_validation)
    X_test = reshape_data(X_test)

    return X_train, X_validation, X_test, y_train, y_validation, y_test


class SampleDataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        x = torch.Tensor(self.x[idx])
        y = torch.as_tensor(np.array(self.y[idx])).long()
        return x, y


class Network(nn.Module):

    def __init__(self):
        super().__init__()

        self.conv_block1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(3, 3)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
            nn.BatchNorm2d(32),
        )

        self.conv_block2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
            nn.BatchNorm2d(32),
        )

        self.conv_block3 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(2, 2)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
            nn.BatchNorm2d(32),
        )

        self.dense_block = nn.Sequential(
            nn.Linear(in_features=480, out_features=64),
            nn.Dropout(0.3),
            nn.Linear(in_features=64, out_features=10),
        )

    def forward(self, x):
        out = self.conv_block1(x)
        out = self.conv_block2(out)
        out = self.conv_block3(out)
        out = out.view(out.size(0), -1)  # Flatten
        out = self.dense_block(out)
        return out


def multiclass_acc(y_pred, y_true):
    _, y_pred_tags = torch.max(y_pred, dim=1)

    correct_pred = (y_pred_tags == y_true).float()
    acc = correct_pred.sum() / len(correct_pred)

    return acc


def plot_history(history):

    fig, axs = plt.subplots(2)

    axs[0].plot(history["train_acc"], label="train accuracy")
    axs[0].plot(history["eval_acc"], label="test accuracy")
    axs[0].set_ylabel("Accuracy")
    axs[0].set_xlabel("Epoch")
    axs[0].legend(loc="lower right")
    axs[0].set_title("Accuracy eval")

    axs[1].plot(history["train_loss"], label="train error")
    axs[1].plot(history["eval_loss"], label="test error")
    axs[1].set_ylabel("Error")
    axs[1].set_xlabel("Epoch")
    axs[1].legend(loc="upper right")
    axs[1].set_title("Error eval")

    plt.show()


if __name__ == "__main__":

    X_train, X_val, X_test, y_train, y_val, y_test = prepare_dataset(0.25, 0.2)
    input_shape = X_train.shape[1:]

    train_dataset = SampleDataset(X_train, y_train)
    val_dataset = SampleDataset(X_val, y_val)
    test_dataset = SampleDataset(X_test, y_test)

    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=True)

    num_epochs = 30
    learning_rate = 0.0001

    model = Network()
    summary(model, input_shape)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=learning_rate,
                                 weight_decay=0.001)

    ls_avg_train_acc = []
    ls_avg_train_loss = []
    ls_avg_eval_acc = []
    ls_avg_eval_loss = []

    for epoch in range(num_epochs):

        train_acc = 0
        train_loss = 0

        model.train()
        for idx, data in enumerate(train_dataloader):
            inputs = data[0]
            train_labels = data[1]

            outputs = model(inputs)
            loss = criterion(outputs, train_labels.squeeze())
            acc = multiclass_acc(outputs, train_labels)

            train_loss += float(loss.item())
            train_acc += float(acc.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        avg_train_acc = train_acc / len(train_dataloader)
        avg_train_loss = train_loss / len(train_dataloader)
        ls_avg_train_acc.append(avg_train_acc)
        ls_avg_train_loss.append(avg_train_loss)

        model.eval()
        with torch.no_grad():

            eval_loss = 0
            eval_acc = 0

            for idx, eval_batch in enumerate(val_dataloader):
                eval_data = eval_batch[0]
                eval_label = eval_batch[1]

                pred = model(eval_data)
                loss = criterion(pred, eval_label.squeeze())

                acc = multiclass_acc(pred, eval_label)

                eval_loss += float(loss)
                eval_acc += float(acc)

            avg_eval_acc = eval_acc / len(val_dataloader)
            avg_eval_loss = eval_loss / len(val_dataloader)
            ls_avg_eval_acc.append(avg_eval_acc)
            ls_avg_eval_loss.append(avg_eval_loss)

        if (epoch + 1) % 5 == 0:
            print("\nEpoch [{}/{}], Acc: {:.4f}, Loss: {:.4f}".format(epoch + 1,
                                                                      num_epochs,
                                                                      avg_train_acc,
                                                                      avg_train_loss))

            print('Evaluation loss: {}'.format(avg_eval_loss))
            print('Evaluation acc: {}'.format(avg_eval_acc))

    history = {"train_acc": ls_avg_train_acc,
               "train_loss": ls_avg_train_loss,
               "eval_acc": ls_avg_eval_acc,
               "eval_loss": ls_avg_eval_loss}

    plot_history(history)


