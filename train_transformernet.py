# Remove warning
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

from argparse import ArgumentParser

import numpy as np
import torch
from torch.utils.data import DataLoader

from transformernet.transformernet import TSP_net
from transformernet.utils import test_transformernet, train_transformernet
from utils import collate_batch_transformernet, split_train_test

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(DEVICE)


def main(args):

    instances = np.load("data/instances.npy")
    instances_orders = np.load("data/instances_orders.npy")

    X, y = instances, instances_orders
    y[y == -1] = 50
    train_dataset, test_dataset = split_train_test(X, y, 0.1)

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.bs,
        shuffle=True,
        collate_fn=collate_batch_transformernet,
    )

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=16,
        shuffle=True,
        collate_fn=collate_batch_transformernet,
    )

    net = TSP_net(2, 128, 512, 6, 2, 8, 1000, False).to(DEVICE)

    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)
    criterion = torch.nn.NLLLoss().to(DEVICE)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, [1, 2, 3], gamma=0.1
    )  # Helps the training

    for epoch in range(1, args.epochs + 1):
        train_transformernet(train_dataloader, net, optimizer, criterion, epoch)
        test_transformernet(test_dataloader, net, criterion, epoch)
        scheduler.step()
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": net.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
            },
            "transformernets/ckpts/" + f"checkpoint_epoch{epoch}.pth",
        )


if __name__ == "__main__":

    parser = ArgumentParser()

    parser.add_argument(
        "--epochs",
        default=5,
        type=int,
        help="number of training epochs",
    )
    parser.add_argument(
        "--lr",
        default=0.01,
        type=float,
        help="starting learning rate",
    )
    parser.add_argument(
        "--bs",
        default=32,
        type=int,
        help="batch size for the training dataloder",
    )

    args = parser.parse_args()

    main(args)
