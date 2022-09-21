from pycsou.core.linop import LinearOperator
import numpy as np
from bs_utils import convolve_coeffs

class bs_PosGridEval(LinearOperator):

    def __init__(self, coeffs_shape, bs_filter, bcs, dtype: type = np.float64):
        # Evaluate spline on all points of grid by convolution of the coefficients with the b-spline filter
        r"""
        Parameters
        ----------
        size : tuple of ints representing the shape of the input array.
        coeffs_shape :
        deriv_filters : list of np.array() filters as [b_spline filter, 1st deriv filter, 2nd deriv filter]
        bcs : list of behavior of the bcs in each axis in relation to the coefficients
        """

        self.coeffs_shape = coeffs_shape
        self.bs_filter = bs_filter
        self.bcs = bcs

        self.adj_bs_filter = bs_filter[::-1]
        self.adj_bcs = self.bcs

        input_size = np.prod(coeffs_shape) # pycsou flattening
        output_size = input_size # pycsou flattening
        super(bs_PosGridEval, self).__init__(shape=(output_size, input_size), dtype=dtype)


    def __call__(self, coeffs_flat: np.ndarray) -> np.ndarray:

        coeffs = coeffs_flat.reshape(self.coeffs_shape) # pycsou reshape
        conv = convolve_coeffs(coeffs, self.bs_filter, self.bcs)
        # may add i+(grideval) directly with corresponding prox
        return conv.ravel() # pycsou flatten


    def adjoint(self, evals_flat: np.ndarray) -> np.ndarray:

        evals = evals_flat.reshape(self.coeffs_shape) # pycsou reshape
        adj_conv = convolve_coeffs(evals, self.adj_bs_filter, self.adj_bcs)
        return adj_conv.ravel() # pycsou flatten
