import numpy as np
from scipy import spatial
from sklearn import metrics
from typing import Union, Callable


def get_pairwise_distances(
    X: np.ndarray, metric: Union[str, Callable] = "euclidean", **kwargs
):
    if isinstance(metric, str):
        if not (
            metric in metrics.pairwise.PAIRED_DISTANCES
            or metric in spatial.distances.__dict__
        ):
            raise ValueError(
                f"Metric {metric} is not a valid pairwise distance metric. Must be one of entries in sklearn.metrics.pairwise.PAIRED_DISTANCES or scipy.spatial.distances"
            )

    elif isinstance(metric, Callable) is False:
        raise TypeError("Argument metric must be either a string or a callable.")

    n_jobs = kwargs.get("n_jobs")
    if n_jobs is not None:
        if isinstance(n_jobs, int) is False:
            raise TypeError(
                "Argument n_jobs to pairwise_distances must be of type int."
            )

    pdist = metrics.pairwise_distances(X, metric=metric, **kwargs)
    pdist[
        np.diag_indices_from(pdist)
    ] = np.inf  # set self-connection weight to infinity
    return metrics.pairwise_distances(X, metric=metric, **kwargs)
