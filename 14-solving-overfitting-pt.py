import json
import numpy as np
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

DATA_PATH = "datasets/processed/data_10.json"


def load_data(data_path):

    with open(data_path, "r") as fp:
        data = json.load(fp)

    X = np.array(data["mfcc"])
    y = np.array(data["labels"])

    return X, y


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

        self.layers = nn.Sequential(
            nn.Linear(in_features=1690, out_features=512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(in_features=512, out_features=256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(in_features=256, out_features=64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(in_features=64, out_features=10),
            # nn.Softmax(dim=-1)
            # softmax is applied only on the data, not on batch thus dim=-1)
            # softmax is already included in nn.CrossEntropy, thus it should be taken out from here
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)
        out = self.layers(x)
        return out


def multiclass_acc(y_pred, y_test):
    _, y_pred_tags = torch.max(y_pred, dim=1)

    correct_pred = (y_pred_tags == y_test).float()
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
    X, y = load_data(DATA_PATH)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

    train_dataset = SampleDataset(X_train, y_train)
    test_dataset = SampleDataset(X_test, y_test)

    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=True)

    num_epochs = 100
    learning_rate = 0.0001

    model = Network()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.001) #L2 regularization

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
            targets = data[1]

            outputs = model(inputs)
            loss = criterion(outputs, targets.squeeze())
            acc = multiclass_acc(outputs, targets)

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

            total_loss = 0
            total_acc = 0
            for idx, eval_batch in enumerate(test_dataloader):
                eval_data = eval_batch[0]
                eval_label = eval_batch[1]

                pred = model(eval_data)
                loss = criterion(pred, eval_label.squeeze())
                acc = multiclass_acc(pred, eval_label)

                total_loss += float(loss)
                total_acc += float(acc)


            avg_eval_acc = total_acc / len(test_dataloader)
            avg_eval_loss = total_loss / len(test_dataloader)
            ls_avg_eval_acc.append(avg_eval_acc)
            ls_avg_eval_loss.append(avg_eval_loss)

        if (epoch + 1) % 5 == 0:
            print("\nEpoch [{}/{}], Acc: {:.4f}, Loss: {:.4f}".format(epoch + 1,
                                                                      num_epochs,
                                                                      avg_train_acc,
                                                                      avg_train_loss))

            print('Evaluation loss: {}'.format(avg_eval_loss))
            print('Evaluation acc: {}'.format(avg_eval_acc))

    history = {"train_acc" : ls_avg_train_acc,
            "train_loss" : ls_avg_train_loss,
            "eval_acc" : ls_avg_eval_acc,
            "eval_loss" : ls_avg_eval_loss}

    plot_history(history)