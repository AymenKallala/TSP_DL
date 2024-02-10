from scipy.spatial import distance_matrix
import numpy as np
from sklearn.model_selection import train_test_split
import torch

DEVICE = "cuda" if torch.cuda.is_available() else  "cpu"


def unpad(array):
    idxs = np.where(array == -1)[0]
    if len(idxs) > 0:
        array[idxs[0]:] = np.arange(idxs[0],50)
    return array

def length(order,distance_matrix):
    length = 0
    idx = 0
    for i in order:
        length+= distance_matrix[idx][i]
        idx = i
    return length

def gap(pred_length,optim_length):
    return 100*(pred_length-optim_length)/optim_length

def evaluate_gap(y_pred,y_true,x):
        dm = distance_matrix(x,x)
        l_pred = length(y_pred,dm)
        l_true = length(y_true,dm)
        return gap(l_pred,l_true)

def split_train_test(X,y,split):
    """Split the dataset into trainset and testset, with each instances linked to its groundtruth solution"""

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=split, random_state=42)
    train_dataset = list(zip(X_train,y_train))
    test_dataset = list(zip(X_test,y_test))
    return train_dataset,test_dataset
def collate_batch_ptr(batch):
    points_list, target_list = [], []

    for x,y in batch:
         
         points_list.append(x)
         target_list.append(y)
    return torch.tensor(np.array(points_list)).to(DEVICE),torch.tensor(np.array(target_list)).to(DEVICE)
