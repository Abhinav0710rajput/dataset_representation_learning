import torch
import random

def sample_batch(X: torch.Tensor, Y: torch.Tensor):
    """
    X: (N, M) predictor matrix
    Y: (N, T) target matrix
    Returns a batch (X', Y') with random subset of instances, predictors, and targets
    """
    N, M = X.shape
    _, T = Y.shape

    N_prime = min( N, 256)      # 16 to 256
    M_prime = min( M, 32) ###########
    T_prime = min(T, 5)


    instance_idx = torch.randperm(N)[:N_prime]
    predictor_idx = torch.randperm(M)[:M_prime]
    target_idx = torch.randperm(T)[:T_prime]

    X_prime = X[instance_idx][:, predictor_idx]
    Y_prime = Y[instance_idx][:, target_idx]

    return X_prime, Y_prime



def sample_batch_pairs(dataset_list):
    """
    dataset_list: list of (X, Y) tuples
    Returns: (X1, Y1), (X2, Y2), label ∈ {0,1}
    """
    idx = random.randint(0, len(dataset_list) - 1)
    D = dataset_list[idx]

    if random.random() < 0.5:
        # Positive pair: two batches from the same dataset
        X1, Y1 = sample_batch(*D)
        X2, Y2 = sample_batch(*D)
        label = 1
    else:
        # Negative pair: two batches from different datasets
        alt_indices = list(range(len(dataset_list)))
        alt_indices.remove(idx)
        idx2 = random.choice(alt_indices)
        D2 = dataset_list[idx2]

        X1, Y1 = sample_batch(*D)
        X2, Y2 = sample_batch(*D2)
        label = 0

    return (X1, Y1), (X2, Y2), label
