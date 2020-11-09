import json
import numpy as np
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader




def read_data(data_path):

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


def setup_dataloader(config):
    X, y = read_data(config.data_path)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=config.test_ratio)

    train_dataset = SampleDataset(X_train, y_train)
    eval_dataset = SampleDataset(X_test, y_test)

    train_dataloader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    eval_dataloader = DataLoader(eval_dataset, batch_size=config.batch_size, shuffle=True)

    return train_dataloader, eval_dataloader