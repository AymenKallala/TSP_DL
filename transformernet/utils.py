import torch
from tqdm import tqdm

from graphconvnet.model.graph_utils import total_tour_len_nodes


def train_transformernet(dataloader, model, optimizer, criterion, epoch):
    model.train()
    total_loss, total_count = (
        0,
        0,
    )
    log_interval = 100
    running_tour_length = 0
    running_gt_length = 0

    for idx, (x, y, mask, dm) in tqdm(enumerate(dataloader)):
        optimizer.zero_grad()
        total_count += y.shape[0]

        y_pred, probs, _ = model(x, mask)
        probs = probs.permute(0, 2, 1)

        # Get the loss
        loss = criterion(probs, y)
        total_loss += loss.item()
        # Do back propagation
        loss.backward()
        # Clip the gradients at 0.1
        # nn.utils.clip_grad_norm_(model.parameters(), 0.1)
        # Do an optimization step
        optimizer.step()

        # Compute the tour lengths
        running_tour_length += total_tour_len_nodes(dm, y_pred)
        running_gt_length += total_tour_len_nodes(dm, y)

        if idx % log_interval == 0 and idx > 0:
            gap = 100 * (running_tour_length - running_gt_length) / running_gt_length
            print(
                "| epoch {:3d} | {:5d}/{:5d} batches"
                "| gap {:8.3f}".format(epoch, idx, len(dataloader), gap),
                f"| loss {total_loss/total_count}",
            )
            running_tour_length, running_gt_length, total_count, total_loss = 0, 0, 0, 0


def test_transformernet(dataloader, net, criterion, epoch):
    """
    Test a graph conv net on validation data

    Args:
        dataloader : testing data
        net : a model instance
        epoch (int): the current epoch of testing
        dtypeFloat : Type variable (for the beamserch call)
        dtypeLong : Type variable (for the beamserch call)
    """

    net.eval()  # set model in eval mode
    log_interval = 10

    # Initialize running data
    running_loss = 0.0
    running_nb_data = 0.0
    running_tour_length = 0.0
    running_gt_length = 0.0

    total_gaps = 0
    total_count = 0

    # Predictions

    with torch.no_grad():
        for idx, (x, y, mask, dm) in tqdm(enumerate(dataloader)):

            # Forward pass
            y_pred, probs, _ = net(x, mask)
            # Get the loss
            loss = criterion(probs, y)
            running_loss += loss.item()

            # Compute the tour lengths
            running_tour_length += total_tour_len_nodes(dm, y_pred)
            running_gt_length += total_tour_len_nodes(dm, y)

            # Update running data
            running_nb_data += len(x)
            running_loss += loss.item()  # Re-scale loss

            if idx % log_interval == 0 and idx > 0:
                gap = (
                    100 * (running_tour_length - running_gt_length) / running_gt_length
                )

                total_gaps += gap
                total_count += 1

                print(
                    "| epoch {:3d} | {:5d}/{:5d} batches".format(
                        epoch, idx, len(dataloader)
                    ),
                    f"| loss {running_loss/running_nb_data}" f"| gap {gap}",
                )
                (
                    running_loss,
                    running_nb_data,
                    running_tour_length,
                    running_gt_length,
                ) = (0, 0, 0, 0)

        print("-" * 50, f"AVERAGE GAP : {total_gaps/total_count}", "-" * 50)
