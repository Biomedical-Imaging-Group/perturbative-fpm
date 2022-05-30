import numpy as np
from utils.compute_err_grad import compute_error_gradient, compute_error, compute_gradient
from utils.eval import eval_vector

def gradient_descent(y, A, x0, x=None, n_iter=100, line_search=None, step_size=10, log=False, renorm=False, verbose=0):
    """
    Gradient descent starting from the initial estimate x0, to solve a phase retrieval problem:
    y = np.abs(A @ x) ** 2
    The optimized cost is the mean-square error.
    """
    if verbose:
        print("Computing gradient descent iteration to solve a phase retrieval problem...")

    # n, d = A.shape
    norm_est = np.sqrt(np.mean(y)) / np.mean(np.sum(A**2, axis=1))  # std of each component of x
    if log:
        corr_vec = np.zeros(n_iter+1)
        cost_vec = np.zeros(n_iter+1)
        corr_vec[0], cost_vec[0] = eval_vector(A, x, x0)

    curr_x = x0
    for i_iter in range(n_iter):
        err, grad = compute_error_gradient(y, A, curr_x)
        # the scale of the gradient is given by A @ (y * A@x) / n
        step = step_size  # / norm_est**2  # modify this line if other line search strategies are used
        curr_x = curr_x - step * grad.view(np.complex128)
        if renorm:
            curr_x *= norm_est / np.sqrt(np.mean(curr_x**2))
        if log:
            corr_vec[i_iter+1], cost_vec[i_iter+1] = eval_vector(A, x, curr_x)

    if verbose:
        print("Gradient descent complete")

    if log:
        return curr_x, corr_vec, cost_vec
    else:
        return curr_x
