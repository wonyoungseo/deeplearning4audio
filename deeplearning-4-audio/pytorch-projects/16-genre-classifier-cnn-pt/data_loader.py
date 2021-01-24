import json
import numpy as np
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader


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


def prepare_dataset(data_path, test_size, validation_size):
    # load data
    X, y = load_data(data_path)

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


def setup_dataloader(config):
    X_train, X_val, X_test, y_train, y_val, y_test = prepare_dataset(config.data_path, 0.25, 0.2)

    train_dataset = SampleDataset(X_train, y_train)
    val_dataset = SampleDataset(X_val, y_val)
    test_dataset = SampleDataset(X_test, y_test)

    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=True)

    return train_dataloader, val_dataloader, test_dataloader