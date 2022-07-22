import numpy as np


def compute_improvement(a, b) -> float:
    if a == 0:
        raise ValueError("first argument can't be 0")
    diff = b - a
    return diff / a


def compute_gini(samples):
    assert len(samples.shape) == 1

    n_samples = samples.shape[0]
    total = samples.sum()
    summation = 0
    for i, ix in enumerate(np.argsort(samples)):

        summation += ((n_samples - i) / (n_samples + 1)) * (samples[ix] / total)

    return 1 - (2 * summation)


def compute_user_gini(binary_matrix):
    return compute_gini(binary_matrix.sum(axis=1).A.T[0])


def compute_item_gini(binary_matrix):
    return compute_gini(binary_matrix.sum(axis=0).A[0])
