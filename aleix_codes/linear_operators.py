from __init__ import array_lib as np
from plt import plt


class matrix_weighting():
    def __init__(self, A):
        self.A = A.ravel()
        self.lip = np.max(A)
        self.eigen_min = np.min(A)
        self.dim = (self.A).size

    def __call__(self, sino):
        return self.A*sino

    def adjoint(self, sino):
        return  self.A.T*sino


class LinOpGrad():
    def __init__(self, shp):
        import pylops as pl #pylops supports both numpy and cupy, pylops_gpu for pytorch

        self.shp = shp
        self.ndim = len(shp)
        self.Grad = pl.Gradient(shp)
    def __call__(self, x):
        return self.Grad.matvec(x) #self.Grad(x)
        
    def adjoint(self, x):        
        return self.Grad.rmatvec(x) #self.Grad.adjoint(x) to create hermitian adjoint