from copy import deepcopy
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from model import Network
from utils import multi_acc


class Trainer:

    def __init__(self,
                 config,
                 train_dataloader,
                 eval_dataloader):

        self.config = config
        self.model = Network(self.config.dropout_rate)

        self.optimizer = optim.Adam(self.model.parameters(),
                                    lr=config.learning_rate,
                                    weight_decay=config.weight_decay)
        self.criterion = nn.CrossEntropyLoss()
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader

        # self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # self.model.to(self.device)

    def train(self):

        self.model.train()

        for epoch in range(self.config.num_epochs):
            train_acc = 0
            train_loss = 0

            for idx, data in enumerate(self.train_dataloader):
                inputs = data[0]
                targets = data[1]

                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets.squeeze())
                acc = multi_acc(outputs, targets.squeeze())

                train_loss += float(loss.item())
                train_acc += float(acc.item())

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            if (epoch + 1) % 5 == 0:
                print("\nEpoch [{}/{}],\nAcc: {:.4f}, Loss: {:.4f}".format(
                    epoch + 1,
                    self.config.num_epochs,
                    train_acc / len(self.train_dataloader),
                    train_loss / len(self.train_dataloader)))

                self.evaluate()


    def evaluate(self):

        self.model.eval()
        with torch.no_grad():

            total_loss = 0
            total_acc = 0

            for idx, eval_batch in enumerate(self.eval_dataloader):
                eval_data = eval_batch[0]
                eval_label = eval_batch[1]

                pred = self.model(eval_data)
                loss = self.criterion(pred, eval_label.squeeze())
                acc = multi_acc(pred, eval_label)

                total_loss += float(loss)
                total_acc += float(acc)


            avg_loss = total_loss / len(self.eval_dataloader)
            avg_acc = total_acc / len(self.eval_dataloader)

            print('Eval Acc: {}, Loss: {}'.format(avg_acc,
                                                  avg_loss))

