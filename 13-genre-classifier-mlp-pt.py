import json
from random import random
import numpy as np
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
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
            nn.Linear(in_features=512, out_features=256),
            nn.ReLU(),
            nn.Linear(in_features=256, out_features=64),
            nn.ReLU(),
            nn.Linear(in_features=64, out_features=10),
            nn.Softmax(dim=-1)  # 소프트맥스는 (bs, hs)에서 hs에만 적용. 각 샘플 별로 소프트맥스 적용
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)
        out = self.layers(x)
        return out


def multi_acc(y_pred, y_test):
    _, y_pred_tags = torch.max(y_pred, dim=1)

    correct_pred = (y_pred_tags == y_test).float()
    acc = correct_pred.sum() / len(correct_pred)
    acc = torch.round(acc)

    return acc

if __name__ == "__main__":
    X, y = load_data(DATA_PATH)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

    train_dataset = SampleDataset(X_train, y_train)
    test_dataset = SampleDataset(X_test, y_test)

    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=True)

    num_epochs = 50
    learning_rate = 0.0001

    model = Network()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):

        train_acc = 0
        train_loss = 0

        ### To-do
        ### fix error on train_dataloader
        model.train()
        for idx, data in enumerate(train_dataloader):
            inputs = data[0]
            targets = data[1]

            outputs = model(inputs)
            loss = criterion(outputs, targets.squeeze())
            acc = multi_acc(outputs, targets)

            train_loss += float(loss.item())
            train_acc += float(acc.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if (epoch + 1) % 5 == 0:
            print("\nEpoch [{}/{}], Acc: {:.4f}, Loss: {:.4f}".format(epoch + 1,
                                                                      num_epochs,
                                                                      train_acc / len(train_dataloader),
                                                                      train_loss / len(train_dataloader)))

            model.eval()
            with torch.no_grad():

                total_loss = 0
                total_acc = 0
                for idx, eval_batch in enumerate(test_dataloader):
                    eval_data = eval_batch[0]
                    eval_label = eval_batch[1]

                    pred = model(eval_data)
                    loss = criterion(pred, eval_label.squeeze())
                    acc = multi_acc(pred, eval_label)

                    total_loss += float(loss)
                    total_acc += float(acc)

                avg_loss = total_loss / len(test_dataloader)
                avg_acc = total_acc / len(test_dataloader)

                print('Evaluation loss: {}'.format(avg_loss))
                print('Evaluation acc: {}'.format(avg_acc))