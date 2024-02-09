from python_tsp.heuristics import solve_tsp_local_search
from scipy.spatial import distance_matrix



def local_search_predict(x):

    dm = distance_matrix(x,x)
    return (solve_tsp_local_search(dm)[0])

class LocalSearch():

    def __init__(self,x):

        self.x = x
        self.distance_matrix = distance_matrix(x,x)
    
    def predict(self):

        return(solve_tsp_local_search(self.distance_matrix)[0])