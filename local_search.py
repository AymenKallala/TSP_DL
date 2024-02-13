from multiprocessing import get_context

import torch
from tqdm import tqdm
from concorde.tsp import TSPSolver

from graphconvnet.model.graph_utils import total_gap


def concorde_solve(x):

    solver = TSPSolver.from_data(x[:,0], x[:,1], norm="GEO")  
    solution = solver.solve(verbose=False)
    return solution.tour



def test_concorde(dataloader):
    """Runs local search on a data loader in a parallelized fashion (with multiprocessing)

    Args:
        dataloader (torch.dataloader): the dataset
    """
    total_gaps = 0
    total = 0

    for x, y, dm in tqdm(dataloader, total=len(dataloader)):
        total += len(y)

        with get_context("spawn").Pool() as pool:
            preds = pool.map(concorde_solve, x)

        #total_pred_len = total_tour_len_nodes(dm, torch.Tensor(preds).int())
        #total_gt_len = total_tour_len_nodes(dm, y)

        total_gaps += total_gap(dm, torch.Tensor(preds).int(),y)

    print(f"AVERAGE GAP FOR LOCAL SEARCH SOLVER: {total_gaps/total}")
