import torch


def multiclass_acc(y_pred, y_true):
    """
    :param y_pred:
        torch tensor shape of (num_samples, num_class), driven from model
    :param y_true:
        torch tensor shape of (num_samples)
    :return:
        float accuracy score
    """

    _, y_pred_tags = torch.max(y_pred, dim=1)

    correct_pred = (y_pred_tags == y_true).float()
    acc = correct_pred.sum() / len(correct_pred)

    return acc