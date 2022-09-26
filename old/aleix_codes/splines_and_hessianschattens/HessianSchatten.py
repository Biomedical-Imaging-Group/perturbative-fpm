from typing import Tuple, Optional, Union
from numbers import Number
import numpy as np
from pycsou.core.functional import ProximableFunctional

def nonneg_indicator_project(x): #or take from pycsou/math/prox
    #non-negative
    y = np.real(x)
    y[y < 0] = 0
    return y

def nonneg_indicator_norm(x): #or take from pycsou/math/prox
    #non-negative
    return 0 if np.any(x<0) else np.infty

def id_indicator_project(x):
    return x

def id_indicator_norm(x):
    return 0

class HessianSchattenNorm(ProximableFunctional):
    r"""
    Class for Hessian Schatten Norm.
    """

    def __init__(self, Hess, SchattenNorm, n_prox_iter, iter_func=None, indicator_project=None, indicator_norm=None, hess_ct=None):
        r"""
        Parameters
        ----------
        Hess: instance of a Hessian class
            acting as a linear operator (with .__call__ and .adjoint and .dim)
        SchattenNorm: instance of the LpSchattenNorm class
        n_prox_iter: int
            Number of iterations for the prox computation
        iter_func: function
            Any rule to update n_prox_iter every time it is called
        indicator_project:
            Projector related to desired indicator function
        indicator_norm:
            Norm of the above
        """
        self.Hess = Hess
        self.SchattenNorm = SchattenNorm
        self.coeffs_shape = Hess.coeffs_shape
        self.hess_shape = Hess.hess_shape
        self.hess_dim = np.prod(self.hess_shape)
        input_size = np.prod(self.coeffs_shape)

        self.n_prox_iter = n_prox_iter
        if iter_func == None: self.iter_func = lambda x: x
        if indicator_project == None: self.indicator_project = lambda x: x
        if indicator_norm == None: self.indicator_norm = lambda x: 0
        if hess_ct == None: self.hess_ct = 16*(self.Hess.dim**2)

        super(HessianSchattenNorm, self).__init__(dim=input_size, data=None, is_differentiable=False, is_linear=False) # is_convex = True # pycsou dim


    def __call__(self, coeffs_flat: Union[Number, np.ndarray]) -> Number:

        return self.SchattenNorm(self.Hess(coeffs_flat)) + self.indicator_norm(coeffs_flat)


    def prox(self, z_flat: Union[Number, np.ndarray], tau: Number) -> Union[Number, np.ndarray]:

        self.n_prox_iter = self.iter_func(self.n_prox_iter)
        ct = 1./(self.hess_ct*tau)
        i=0
        Psi = Omega_prev = np.zeros(self.hess_dim)
        t_prev = 1

        while i<self.n_prox_iter:

            Omega = self.SchattenNorm.project(
                    Psi + ct*self.Hess(
                    self.indicator_project(
                    z_flat-tau*self.Hess.adjoint(Psi))), radius=1)
            t = 0.5*(1 + np.sqrt(1+4*t_prev**2))
            Psi = Omega + (Omega - Omega_prev)*(t_prev-1)/t

            i=i+1
            t_prev = t
            Omega_prev = Omega

        return self.indicator_project(z_flat-tau*self.Hess.adjoint(Omega))
