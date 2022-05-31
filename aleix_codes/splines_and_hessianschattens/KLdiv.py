from pycsou.core.functional import DifferentiableFunctional#, ProximableFunctional
from pycsou.func import KLDivergence

from typing import Union
from numbers import Number
import numpy as np


def KLdiv(dim, data, beta=0):
    if beta>0:
        return diff_KLDivergence(dim=dim, data=data, beta=beta)
    else:
        return KLDivergence(dim=dim, data=data)


class diff_KLDivergence(DifferentiableFunctional): # convex, separable

    def __init__(self, dim: int, data: Union[Number, np.ndarray], beta: Number):

        self.grad_lips = np.max(data)/(beta**2)
        super(diff_KLDivergence, self).__init__(dim=dim, data=None, is_linear=False,
                                           diff_lipschitz_cst=self.grad_lips)
        self.data = data
        self.beta = beta


    def __call__(self, x: Union[Number, np.ndarray]) -> Number:

        with np.errstate(divide='ignore'):
            return np.sum(x-self.data*np.log(x+self.beta))


    def jacobianT(self, x: Union[Number, np.ndarray]) -> Union[Number, np.ndarray]:

        return 1-self.data/(x+self.beta)


    def prox(self, x: Union[Number, np.ndarray], tau: Number) -> Union[Number, np.ndarray]:

        d = (x-tau-self.beta)**2 + 4*(x*self.beta + tau*(self.data-self.beta))
        pr = np.zeros_like(x)
        nneg = d>=0
        pr[nneg] = (x[nneg] - tau - self.beta + np.sqrt(d[nneg]))/2.
        return pr
