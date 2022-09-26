import numpy as np

def compute_error(y, A, x):
    """
    Computes the mean-square error of the phase retrieval problem:
    y = np.abs(A @ x) ** 2
    where y contains the measured intensities, A is the measurement matrix and x our current estimate
    """
    n = A.shape[0]

    field = A @ x
    est_y = np.abs(field) ** 2
    return np.mean((est_y - y) ** 2)


def compute_gradient(y, A, x):
    """
    Computes the gradient of the MSE of the phase retrieval problem
    (see compute_error)
    """
    n = A.shape[0]

    field = A @ x
    est_y = np.abs(field) ** 2
    grad = A.T.conj() @ ((est_y - y) * field) / n
    return grad.view(np.double)  # real values for scipy libraries

def compute_error_gradient(y, A, x):
    """
    Computes the MSE and its gradient of the phase retrieval problem
    (see compute_error)
    """
    n = A.shape[0]

    field = A @ x
    est_y = np.abs(field) ** 2
    err = np.mean((est_y - y) ** 2)
    grad = A.astype(np.complex128).T.conj() @ ((est_y - y) * field) / n
    # TOCHECK: why do we need the astype(np.complex128) in the line above?

    return err, grad.view(np.double)

