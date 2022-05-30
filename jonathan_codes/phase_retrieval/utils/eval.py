import numpy as np

# The following function will be very useful to measure performance
def eval_vector(A, x0, x=None, y=None):
    # Compute current output
    y0 = np.abs(A.dot(x0))**2

    if x is not None:
        y = np.abs(A.dot(x))**2
        # The correlation is the scalar product between the true x and the estimate x0
        corr = np.abs(np.conj(x).T @ x0) / (np.linalg.norm(x)*np.linalg.norm(x0))
        corr = np.ravel(corr)[0]
        # The cost is the l2 error on the intensity
        cost = np.linalg.norm(y-y0) / np.linalg.norm(y)
        return corr, cost
    else:
        # The cost is the l2 error on the intensity
        cost = np.linalg.norm(y-y0) / np.linalg.norm(y)
        return cost

def eval_matrix(A, X0, X=None, Y=None):
    k = X0.shape[1]
    corr_mat = np.zeros((k, ))
    cost_mat = np.zeros((k, ))
    for i in range(k):
        x0 = X0[:, i]
        if X is not None:
            x = X[:, i]
            corr, cost = eval_vector(A, x0, x)
            corr_mat[i] = corr
            cost_mat[i] = cost
        else:
            y = Y[:, i]
            cost = eval_vector(A, x0, y=y)
            cost_mat[i] = cost
    if X is not None:
        return corr_mat, cost_mat
    else:
        return cost_mat