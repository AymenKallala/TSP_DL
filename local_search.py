from multiprocessing import get_context

import torch
from python_tsp.heuristics import solve_tsp_local_search
from tqdm import tqdm

from graphconvnet.model.graph_utils import total_tour_len_nodes
from utils import gap


def local_search_predict(dm):

    return solve_tsp_local_search(dm)[0]


def test_local_search(dataloader):
    """Runs local search on a data loader in a parallelized fashion (with multiprocessing)

    Args:
        dataloader (torch.dataloader): the dataset
    """
    total_gaps = 0
    total = 0

    for _, y, dm in tqdm(dataloader, total=len(dataloader)):
        total += len(y)

        with get_context("spawn").Pool() as pool:
            preds = pool.map(local_search_predict, dm)

        total_pred_len = total_tour_len_nodes(dm, torch.Tensor(preds).int())
        total_gt_len = total_tour_len_nodes(dm, y)

        total_gaps += gap(total_pred_len, total_gt_len)

    print(f"AVERAGE GAP FOR LOCAL SEARCH SOLVER: {total_gaps/total}")
