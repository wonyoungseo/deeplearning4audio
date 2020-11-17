import argparse
from trainer import Trainer
from data_loader import setup_dataloader


def define_argparser():
    p = argparse.ArgumentParser()

    p.add_argument("--data_path", type=str, default="../../datasets/processed/data_10.json")
    p.add_argument("--test_ratio", type=float, default=0.3)
    p.add_argument("--batch_size", type=float, default=32)
    p.add_argument("--num_epochs", type=int, default=50)
    p.add_argument("--learning_rate", type=float, default=0.0001)
    p.add_argument("--dropout_rate", type=float, default=0.3)
    p.add_argument("--weight_decay", type=float, default=0.001)

    config = p.parse_args()
    return config


def main(config):

    train_dataloader, eval_dataloader = setup_dataloader(config)

    trainer = Trainer(config,
                      train_dataloader,
                      eval_dataloader)

    trainer.train()


if __name__ == "__main__":
    config = define_argparser()
    main(config)

