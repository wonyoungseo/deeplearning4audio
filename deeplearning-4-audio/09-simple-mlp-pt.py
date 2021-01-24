from random import random
import numpy as np
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim
from torch.utils.data import Dataset, DataLoader


# generate dataset
def generate_dataset(num_samples: int, test_size: float) -> np.array:
    x = np.array([[random() / 2 for _ in range(2)] for _ in range(num_samples)])
    y = np.array([[i[0] + i[1]] for i in x])
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size)
    return x_train, x_test, y_train, y_test

# define dataset
class SampleDataset(Dataset):

    def __init__(self, x, y):
        # assert len(x) == len(y)
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        x = torch.Tensor(self.x[idx])
        y = torch.Tensor(self.y[idx])
        return x, y

# define model
class Net(nn.Module):

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(in_features=2, out_features=5)
        self.fc2 = nn.Linear(in_features=5, out_features=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.fc1(x)
        x = self.sigmoid(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x

if __name__ == "__main__":
    x_train, x_test, y_train, y_test = generate_dataset(5000, 0.2)
    print("x_train : {}".format(x_train.shape))
    print("y_train : {}".format(y_train.shape))
    print("x_text : {}".format(x_test.shape))
    print("y_text : {}".format(y_test.shape))
    print('\n')
    train_dataset = SampleDataset(x_train, y_train)
    test_dataset = SampleDataset(x_test, y_test)

    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=31, shuffle=True)

    num_epochs = 100
    learning_rate = 0.1
    model = Net()
    criterion = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        for idx, data in enumerate(train_dataloader):

            inputs = data[0]
            targets = data[1]

            # forward propagation
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            # backward propagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if (epoch + 1) % 10 == 0:
            print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch + 1, num_epochs, loss.item()))

    # evaluate model
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

    # make prediction
    data = np.array([[0.1, 0.2],
                     [0.2, 0.3]])
    data = torch.from_numpy(data).float()
    predictions = model(data)

    print("\nSome predictions:")
    for d, p in zip(data, predictions):
        print("{} + {} = {}".format(d[0], d[1], p[0]))







