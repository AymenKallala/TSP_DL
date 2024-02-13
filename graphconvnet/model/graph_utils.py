import numpy as np
import torch
import torch.nn.functional as F


def tour_nodes_to_W(nodes):
    """Helper function to convert ordered list of tour nodes to edge adjacency matrix."""
    W = np.zeros((len(nodes), len(nodes)))
    for idx in range(len(nodes) - 1):
        i = int(nodes[idx])
        j = int(nodes[idx + 1])
        W[i][j] = 1
        W[j][i] = 1
    # Add final connection of tour in edge target
    W[j][int(nodes[0])] = 1
    W[int(nodes[0])][j] = 1
    return W


def tour_nodes_to_tour_len(nodes, W_values):
    """Helper function to calculate tour length from ordered list of tour nodes."""
    tour_len = 0
    for idx in range(len(nodes) - 1):
        i = nodes[idx]
        j = nodes[idx + 1]
        tour_len += W_values[i][j]
    # Add final connection of tour in edge target
    tour_len += W_values[j][nodes[0]]
    return tour_len


def W_to_tour_len(W, W_values):
    """Helper function to calculate tour length from edge adjacency matrix."""
    tour_len = 0
    for i in range(W.shape[0]):
        for j in range(W.shape[1]):
            if W[i][j] == 1:
                tour_len += W_values[i][j]
    tour_len /= 2  # Divide by 2 because adjacency matrices are symmetric
    return tour_len


def is_valid_tour(nodes, num_nodes):
    """Sanity check: tour visits all nodes given."""
    return sorted(nodes) == [i for i in range(num_nodes)]

def gap(pred_length, optim_length):
    """ return the gap between a predicted tour length and the groundtruth tour length.

    Args:
        pred_length (float): predicted tour length
        optim_length (float): optimal tour length

    Returns:
        float: the gap computed
    """
    return 100 * (pred_length - optim_length) / optim_length

def total_gap(distance_matrix, bs_nodes,gt_nodes):
    """
    Computes total tour length for given batch prediction as node ordering after beamsearch (for Pytorch tensors).

    Args:
        distance_matrix: Edge values (distance) matrix (batch_size, num_nodes, num_nodes)
        bs_nodes: Node orderings (batch_size, num_nodes)

    Returns:
        mean_tour_len: Mean tour length over batch
    """
    y = bs_nodes.cpu().numpy()
    gt_nodes = gt_nodes.cpu().numpy()
    W_val = distance_matrix.cpu().numpy()

    running_gap = 0
    for batch_idx in range(y.shape[0]):
        pred_tour = 0
        gt_tour = 0
        for y_idx in range(y[batch_idx].shape[0] - 1):
            i = y[batch_idx][y_idx]
            j = y[batch_idx][y_idx + 1]
            h = gt_nodes[batch_idx][y_idx]
            k = gt_nodes[batch_idx][y_idx+1]
            pred_tour += W_val[batch_idx][i][j]
            gt_tour += W_val[batch_idx][h][k]
        pred_tour += W_val[batch_idx][j][0]
        gt_tour += W_val[batch_idx][k][0] # Add final connection to tour/cycle

        running_gap+= gap(pred_tour,gt_tour)

    return running_gap
