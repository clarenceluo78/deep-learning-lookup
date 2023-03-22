import numpy as np

def silhouette_coef(X, y_pred):
    """Calculate Silhouette coefficient

    Args:
        X (_type_): cluster sample
        y_pred (_type_): cluster result labels

    Returns:
        _type_: Silhouette coefficient for sample X
    """
    K = len(np.unique(y_pred))
    s_arr = np.zeros(X.shape[0])
    # Go through one sample at a time
    for sample_idx in range(0, X.shape[0]):
        current_sample = X[sample_idx]
        current_cluster = y_pred[sample_idx]
        a = 0
        a_numerator = 0
        a_denominator = 0

        """calculate a"""
        for idx_1 in range(0, X.shape[0]):
            # If the other sample is in the same cluster as this sample
            if current_cluster == y_pred[idx_1]:
                distance = np.linalg.norm(current_sample - X[idx_1])
                a_numerator += distance
                a_denominator += 1
        # Calculate the value for a
        if a_denominator > 0:
            a = a_numerator / a_denominator

        """calculate b"""
        b = float("inf")  # first set to a large upper bound
        for cluster_idx in range(0, K):

            b_numerator = 0
            b_denominator = 0
            b_avg_distance = 0

            if cluster_idx+1 != current_cluster:
                for idx_2 in range(0, X.shape[0]):
                    if y_pred[idx_2] == cluster_idx+1:
                        distance = np.linalg.norm(current_sample - X[idx_2])
                        b_numerator += distance
                        b_denominator += 1
                if b_denominator > 0:
                    b_avg_distance = b_numerator / b_denominator

                # Update b if we have a new minimum
                if b_avg_distance < b:
                    b = b_avg_distance

        # Calculate the Silhouette Coefficient s where s = (b-a)/max(a,b)
        s = (b - a) / max(a,b)
        s_arr[sample_idx] = s

    return np.mean(s_arr)



