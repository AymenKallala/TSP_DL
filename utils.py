import numpy as np
import torch
from scipy.spatial import distance_matrix
from sklearn.model_selection import train_test_split

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def unpad(array):
    idxs = np.where(array == -1)[0]
    if len(idxs) > 0:
        array[idxs[0] :] = np.arange(idxs[0], 50)
    return array


def gap(pred_length, optim_length):
    """ return the gap between a predicted tour length and the groundtruth tour length.

    Args:
        pred_length (float): predicted tour length
        optim_length (float): optimal tour length

    Returns:
        float: the gap computed
    """
    return 100 * (pred_length - optim_length) / optim_length


def split_train_test(X, y, split):
    """Split the dataset into trainset and testset, with each instances linked to its groundtruth solution

    Args:
        X (list): Input list
        y (list): Target list
        split (float): ratio of test size desired

    Returns:
        list, list:  tuple of trainset and testset.
    """

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=split, random_state=42
    )
    train_dataset = list(zip(X_train, y_train))
    test_dataset = list(zip(X_test, y_test))
    return train_dataset, test_dataset


def check_finished_routes(tensor):
    new = []
    for row in tensor:
        new.append(torch.tensor([False in row]))

    return ~torch.stack(new).to(tensor.device)


def collate_batch_local_search(batch):
    """organize the batching for local search model

    Args:
        batch (list): input batch elements

    Returns:
        tensor,tensor,tensor: organized batch with computed distance matrices
    """
    points_list, target_list, distance_matrices = [], [], []

    for x, y in batch:

        points_list.append(x)
        target_list.append(y)
        distance_matrices.append(distance_matrix(x, x))
    return (
        torch.tensor(np.array(points_list)).to("cpu"),
        torch.tensor(np.array(target_list)).to("cpu"),
        torch.tensor(np.array(distance_matrices)).to("cpu"),
    )


def collate_batch_gcn(batch):
    """The necessary modifications to feed the data in the right format to the Graph Conv Net."""
    x_edges = []
    x_edges_values = []
    x_nodes = []
    x_nodes_coord = []
    y_edges = []
    y_nodes = []

    for x, y in batch:
        W = np.ones((len(y), len(y)), dtype=np.int64)
        np.fill_diagonal(W, 2)

        x_nodes_coord.append(x)
        x_nodes.append(np.ones(len(y), dtype=np.int64))
        y_nodes.append(y)
        x_edges.append(W)  # Graph is fully connected
        x_edges_values.append(distance_matrix(x, x))

        # Compute the adjacency matrix
        edges_target = np.zeros((len(y), len(y)), dtype=np.int64)
        for idx in range(len(y) - 1):
            i = y[idx]
            j = y[idx + 1]
            edges_target[i][j] = 1
        edges_target[y[-1]][y[0]] = 1

        y_edges.append(edges_target)

    return (
        torch.LongTensor(x_edges).to(DEVICE),
        torch.FloatTensor(x_edges_values).to(DEVICE),
        torch.LongTensor(x_nodes).to(DEVICE),
        torch.FloatTensor(x_nodes_coord).to(DEVICE),
        torch.LongTensor(y_edges).to(DEVICE),
        torch.LongTensor(y_nodes).to(DEVICE),
    )


def collate_batch_transformernet(batch):
    points_list, target_list, dm = [], [], []

    for x, y in batch:
        dm_x = np.append(x, np.array([[-1, -1]]), axis=0)

        points_list.append(x)
        target_list.append(y)
        dm.append(distance_matrix(dm_x, dm_x))

    x = torch.tensor(np.array(points_list)).to(DEVICE)
    y = torch.tensor(np.array(target_list))
    starting_mask = y == -1
    added_node = torch.zeros(len(x), 1).bool()

    return (
        x,
        y.to(DEVICE),
        torch.cat([starting_mask, added_node], 1).to(x.device),
        torch.FloatTensor(dm).to(x.device),
    )
