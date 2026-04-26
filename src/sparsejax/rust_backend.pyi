import numpy as np

def takahashi(
    indptr: np.ndarray, indices: np.ndarray, data: np.ndarray, perm: np.ndarray, n: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]: ...
def takahashi_masked(
    indptr: np.ndarray,
    indices: np.ndarray,
    data: np.ndarray,
    perm: np.ndarray,
    query_rows: np.ndarray,
    query_cols: np.ndarray,
    n: int,
) -> np.ndarray: ...
