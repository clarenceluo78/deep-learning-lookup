import numpy as np 

def rand_index(y, y_pred):
    """Rand index

    Args:
        y (_type_): true labels
        y_pred (_type_): predict labels

    Returns:
        _type_: value of rand index
    """
    tp_plus_fp = comb(np.bincount(y), 2).sum()
    tp_plus_fn = comb(np.bincount(y_pred), 2).sum()

    A = np.c_[(y, y_pred)]

    # calculate each component
    tp = sum(comb(np.bincount(A[A[:, 0] == i, 1]), 2).sum()
             for i in set(y))
    fp = tp_plus_fp - tp
    fn = tp_plus_fn - tp
    tn = comb(len(A), 2) - tp - fp - fn

    return (tp + tn) / (tp + fp + fn + tn)