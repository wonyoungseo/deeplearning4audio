
import torch









def multi_acc(y_pred, y_test):
    _, y_pred_tags = torch.max(y_pred, dim=1)

    correct_pred = (y_pred_tags == y_test).float()
    acc = correct_pred.sum() / len(correct_pred)
    acc = torch.round(acc)

    return acc