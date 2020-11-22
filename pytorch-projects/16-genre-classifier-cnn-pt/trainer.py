import torch
import torch.nn as nn
import torch.optim as optim
from model import Network
from utils import multiclass_acc
from torchsummary import summary


class Trainer:

    def __init__(self,
                 config,
                 train_dataloader,
                 val_dataloader):

        self.config = config
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader

        self.model = Network(self.config.dropout_rate)
        self.optimizer = optim.Adam(self.model.parameters(),
                                    lr=config.learning_rate,
                                    weight_decay=config.weight_decay)
        self.criterion = nn.CrossEntropyLoss()

    def train(self):

        self.model.train()

        for epoch in range(self.config.num_epochs):
            train_acc = 0
            train_loss = 0

            for idx, batch in enumerate(self.train_dataloader):
                inputs = batch[0]
                train_labels = batch[1]

                outputs = self.model(inputs)
                loss = self.criterion(outputs, train_labels.squeeze())
                acc = multiclass_acc(outputs, train_labels)

                train_loss += float(loss.item())
                train_acc += float(acc.item())

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            if (epoch + 1) % 5 == 0:
                print("\nEpoch [{}/{}], \nAcc: {:.4f}, Loss: {:.4f}".format(
                    epoch+1,
                    self.config.num_epochs,
                    train_acc / len(self.train_dataloader),
                    train_loss / len(self.train_dataloader)
                ))

                self.evaluate()


    def evaluate(self):

        self.model.eval()
        with torch.no_grad():

            eval_loss = 0
            eval_acc = 0

            for idx, val_batch in enumerate(self.val_dataloader):

                val_inputs = val_batch[0]
                val_labels = val_batch[1]

                pred = self.model(val_inputs)
                loss = self.criterion(pred, val_labels.squeeze())
                acc = multiclass_acc(pred, val_labels)

                eval_loss += float(loss)
                eval_acc += float(acc)

            avg_loss = eval_loss / len(self.val_dataloader)
            avg_acc = eval_acc / len(self.val_dataloader)

            print("Eval Acc: {}, Loss: {}".format(avg_acc,
                                                  avg_loss))



