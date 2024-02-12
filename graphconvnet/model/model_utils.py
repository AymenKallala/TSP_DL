import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.utils.class_weight import compute_class_weight
from tqdm import tqdm

from .beamsearch import *
from .graph_utils import *


def train_gcn(dataloader, net, optimizer, epoch):
    """
    Train a graph conv net for one epoch

    Args:
        dataloader : training data
        net : a model instance
        optimizer : optimizer to perform weights updating
        epoch (int): the current epoch of training
    """
    # Set training mode
    net.train()
    # Initially set loss class weights as None
    edge_cw = None
    log_interval = 100

    # Initialize running data
    running_loss = 0.0
    running_nb_data = 0.0

    for idx, batch in tqdm(enumerate(dataloader)):

        x_edges, x_edges_values, x_nodes, x_nodes_coord, y_edges, y_nodes = batch

        # Compute class balance weights (if uncomputed)
        if type(edge_cw) != torch.Tensor:
            edge_labels = y_edges.cpu().numpy().flatten()
            edge_cw = compute_class_weight(
                "balanced", classes=np.unique(edge_labels), y=edge_labels
            )

        # Forward pass
        y_preds, loss = net.forward(
            x_edges, x_edges_values, x_nodes, x_nodes_coord, y_edges, edge_cw
        )
        loss = loss.mean()
        loss.backward()

        # Backward pass
        optimizer.step()
        optimizer.zero_grad()

        # Update running data
        running_nb_data += len(batch)
        running_loss += len(batch) * loss.item()  # Re-scale loss

        # Printing evaluation metrics
        if idx % log_interval == 0 and idx > 0:
            print(
                "| epoch {:3d} | {:5d}/{:5d} batches".format(
                    epoch, idx, len(dataloader)
                ),
                f"| loss {running_loss/running_nb_data}",
            )
            running_loss, running_nb_data = 0, 0


def test_gcn(dataloader, net, epoch, dtypeFloat, dtypeLong):
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
    edge_cw = None
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
        for idx, batch in tqdm(enumerate(dataloader)):
            x_edges, x_edges_values, x_nodes, x_nodes_coord, y_edges, y_nodes = batch

            if type(edge_cw) != torch.Tensor:
                edge_labels = y_edges.cpu().numpy().flatten()
                edge_cw = compute_class_weight(
                    "balanced", classes=np.unique(edge_labels), y=edge_labels
                )

            # Forward pass
            bs_seq, loss = net.decode(
                x_edges, x_edges_values, x_nodes, x_nodes_coord, y_edges, edge_cw
            )
            loss = loss.mean()

            # Beamsearch decoding

            pred_tour_len = total_tour_len_nodes(x_edges_values, bs_seq)
            gt_tour_len = total_tour_len_nodes(x_edges_values, y_nodes)

            # Update running data
            running_nb_data += len(batch)
            running_loss += len(batch) * loss.item()  # Re-scale loss
            running_tour_length += pred_tour_len
            running_gt_length += gt_tour_len

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


def beamsearch_tour_nodes_shortest(
    y_pred_edges,
    x_edges_values,
    beam_size,
    batch_size,
    num_nodes,
    dtypeFloat,
    dtypeLong,
    probs_type="raw",
    random_start=False,
):
    """
    Performs beamsearch procedure on edge prediction matrices and returns possible TSP tours.

    Final predicted tour is the one with the shortest tour length.
    (Standard beamsearch returns the one with the highest probability and does not take length into account.)

    Args:
        y_pred_edges: Predictions for edges (batch_size, num_nodes, num_nodes)
        x_edges_values: Input edge distance matrix (batch_size, num_nodes, num_nodes)
        beam_size: Beam size
        batch_size: Batch size
        num_nodes: Number of nodes in TSP tours
        dtypeFloat: Float data type (for GPU/CPU compatibility)
        dtypeLong: Long data type (for GPU/CPU compatibility)
        probs_type: Type of probability values being handled by beamsearch (either 'raw'/'logits'/'argmax'(TODO))
        random_start: Flag for using fixed (at node 0) vs. random starting points for beamsearch

    Returns:
        shortest_tours: TSP tours in terms of node ordering (batch_size, num_nodes)

    """
    if probs_type == "raw":
        # Compute softmax over edge prediction matrix
        y = F.softmax(y_pred_edges, dim=3)  # B x V x V x voc_edges
        # Consider the second dimension only
        y = y[:, :, :, 1]  # B x V x V
    elif probs_type == "logits":
        # Compute logits over edge prediction matrix
        y = F.log_softmax(y_pred_edges, dim=3)  # B x V x V x voc_edges
        # Consider the second dimension only
        y = y[:, :, :, 1]  # B x V x V
        y[y == 0] = -1e-20  # Set 0s (i.e. log(1)s) to very small negative number
    # Perform beamsearch
    beamsearch = Beamsearch(
        beam_size,
        batch_size,
        num_nodes,
        dtypeFloat,
        dtypeLong,
        probs_type,
        random_start,
    )
    trans_probs = y.gather(1, beamsearch.get_current_state())
    for step in range(num_nodes - 1):
        beamsearch.advance(trans_probs)
        trans_probs = y.gather(1, beamsearch.get_current_state())
    # Initially assign shortest_tours as most probable tours i.e. standard beamsearch
    ends = torch.zeros(batch_size, 1).type(dtypeLong)
    shortest_tours = beamsearch.get_hypothesis(ends)
    # Compute current tour lengths
    shortest_lens = [1e6] * len(shortest_tours)
    for idx in range(len(shortest_tours)):
        shortest_lens[idx] = tour_nodes_to_tour_len(
            shortest_tours[idx].cpu().numpy(), x_edges_values[idx].cpu().numpy()
        )
    # Iterate over all positions in beam (except position 0 --> highest probability)
    for pos in range(1, beam_size):
        ends = pos * torch.ones(batch_size, 1).type(dtypeLong)  # New positions
        hyp_tours = beamsearch.get_hypothesis(ends)
        for idx in range(len(hyp_tours)):
            hyp_nodes = hyp_tours[idx].cpu().numpy()
            hyp_len = tour_nodes_to_tour_len(
                hyp_nodes, x_edges_values[idx].cpu().numpy()
            )
            # Replace tour in shortest_tours if new length is shorter than current best
            if hyp_len < shortest_lens[idx] and is_valid_tour(hyp_nodes, num_nodes):
                shortest_tours[idx] = hyp_tours[idx]
                shortest_lens[idx] = hyp_len
    return shortest_tours
