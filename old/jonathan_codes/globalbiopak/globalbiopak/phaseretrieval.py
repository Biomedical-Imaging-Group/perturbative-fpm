import torch
import numpy as np

class ForwardPhaseRetrieval:
    def __init__(self, linop):
        self.linop = linop

    def apply(self, x):
        return torch.abs(self.linop.apply(x))**2

    def spectralinit(self, y, n_iter=100, method="Lu"):
        """
        Uses the spectral methods to obtain an initial guess of a random Phase Retrieval problem
        The output is the leading eigenvector of a weighted covariance matrix, computed using power iterations
        For more info, see Montanari & Mondelli, 2017
        
        Parameters:
        - y: intensity measurements
        - n_iter: number of power iterations
        
        Output:
        - x_est: final estimate of the spectral method
        """
        
        d = self.linop.in_size
        y_normalized = y / torch.mean(y)  # Normalize intensities
        if method == "Lu":  # if we choose the optimal spectral method by Lu & Li, 2019
            t = torch.maximum(1-1/y_normalized, torch.tensor(-1))  # add a threshold at -5 to avoid very large eigenvalues
        else:  # otherwise we use the method from the Wirtinger flow paper, Candès et al, 2015
            t = y_normalized

        # Power iterations
        x_est = torch.randn(d, dtype=torch.complex64)
        # We first do 10 iterations to detect possible negative eigenvalues
        for i_iter in range(np.minimum(n_iter, 10)):
            # We do not construct the weighted covariance matrix but apply it repeatedly
            x_new = self.linop.apply(x_est)
            x_new = t * x_new
            x_new = self.linop.applyAdjoint(x_new)
            x_est = x_new / torch.norm(x_new)

        # Test if it's a negative eigenvalue
        x_new = self.linop.apply(x_est)
        x_new = t * x_new
        x_new = self.linop.applyAdjoint(x_new)
        corr = torch.real(x_new.ravel().T.conj() @ x_est.ravel())
        if corr < 0:  # if there is a negative eigenvalue, add a regularization
            for i_iter in range(n_iter):
                x_new = self.linop.apply(x_est)
                x_new = t * x_new
                x_new = self.linop.applyAdjoint(x_new)
                x_new = x_new + 1.1*torch.abs(corr)*x_est
                x_est = x_new / torch.norm(x_new)
        else:  # otherwise, finish the power iterations
            for i_iter in range(n_iter - 10):
                x_new = self.linop.apply(x_est)
                x_new = t * x_new
                x_new = self.linop.applyAdjoint(x_new)
                x_est = x_new / torch.norm(x_new)
        return x_est
    
    