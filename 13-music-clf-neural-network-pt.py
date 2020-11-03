import json
from random import random
import numpy as np
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim
from torch.utils.data import Dataset, DataLoader

#### Todos
# 1. Improve model to match performance with TF counterpart module.
# 2. Modify in a more Pythonic, Pytorch style.


DATA_PATH = "datasets/processed/data_10.json"


def load_data(data_path):

    with open(data_path, "r") as fp:
        data = json.load(fp)

    X = np.array(data["mfcc"])
    y = np.array(data["labels"])

    print("Data successfully loaded!")

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
        self.fc1 = nn.Linear(in_features=1690, out_features=512)
        self.fc2 = nn.Linear(in_features=512, out_features=256)
        self.fc3 = nn.Linear(in_features=256, out_features=64)
        self.fc4 = nn.Linear(in_features=64, out_features=10)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax()


    def forward(self, x):
        x = x.view(-1, 1690)

        x = self.fc1(x)
        x = self.relu(x)

        x = self.fc2(x)
        x = self.relu(x)

        x = self.fc3(x)
        x = self.relu(x)

        x = self.fc4(x)
        x = self.softmax(x)
        return x

if __name__ == "__main__":
    X, y = load_data(DATA_PATH)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

    train_dataset = SampleDataset(X_train, y_train)
    test_dataset = SampleDataset(X_test, y_test)

    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=True)

    num_epochs = 50
    learning_rate = 0.001

    model = Network()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):

        ### To-do
        ### fix error on train_dataloader
        for idx, data in enumerate(train_dataloader):
            inputs = data[0]
            targets = data[1]

            outputs = model(inputs)
            loss = criterion(outputs, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if (epoch + 1) % 5 == 0:
            print("Epoch [{}/{}], Loss: {:.4f}".format(epoch + 1,
                                                       num_epochs,
                                                       loss.item()))

    with torch.no_grad():

        total_loss = 0
        for idx, eval_batch in enumerate(test_dataloader):
            eval_data = eval_batch[0]
            eval_label = eval_batch[1]

            pred = model(eval_data)
            loss = criterion(pred, eval_label)

            total_loss += loss

        avg_loss = total_loss / (idx + 1)

        print('\nEvaluation loss: {}'.format(avg_loss))
