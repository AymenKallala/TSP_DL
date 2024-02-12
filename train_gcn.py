# Remove warning
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

from argparse import ArgumentParser

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from graphconvnet.model.gcn_model import ResidualGatedGCNModel
from graphconvnet.model.graph_utils import *
from graphconvnet.model.model_utils import test_gcn, train_gcn
from utils import collate_batch_gcn, split_train_test

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(DEVICE)


# variables

dtypeFloat = torch.cuda.FloatTensor
dtypeLong = torch.cuda.LongTensor


def main(args):

    instances = np.load("data/instances.npy")
    instances_orders = np.load("data/instances_orders_unpadded.npy")

    X, y = instances, instances_orders
    train_dataset, test_dataset = split_train_test(X, y, 0.1)

    train_dataloader = DataLoader(
        train_dataset, batch_size=args.bs, shuffle=True, collate_fn=collate_batch_gcn
    )

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=10,  # Cannot be larger due to memory concerns (computing beamsearch while generating final preds uses a lot)
        shuffle=True,
        collate_fn=collate_batch_gcn,
    )

    net = ResidualGatedGCNModel(dtypeFloat, dtypeLong).to(DEVICE)
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, [1, 2, 3], gamma=0.1
    )  # Helps the training

    for epoch in range(1, args.epochs + 1):
        train_gcn(train_dataloader, net, optimizer, epoch)
        test_gcn(test_dataloader, net, epoch, dtypeFloat, dtypeLong)
        scheduler.step()
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": net.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
            },
            "graphconvnet/ckpts/" + f"checkpoint_epoch{epoch}.pth",
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
