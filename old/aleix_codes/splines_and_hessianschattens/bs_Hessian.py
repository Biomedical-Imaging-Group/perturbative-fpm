from pycsou.core.linop import LinearOperator
from bs_utils import convolve_coeffs, adjoint_conv_bcs
import numpy as np
import itertools as it

def deriv_combinations(dim, deriv_order = 2):
    # could be faster but not expecting high dimensions, otherwise check:
    return np.array([ i for i in it.product(np.arange(deriv_order+1), repeat=dim)
                        if sum(i)==deriv_order])#[::-1]



class bs_Hessian(LinearOperator):

    def __init__(self, coeffs_shape, deriv_filters, bcs, dtype: type = np.float64):
        # Compute Hessian at uniform grid given the coefficients of a separable basis that can be expressed as a convolution
        r"""
        Parameters
        ----------
        size : tuple of ints representing the shape of the input array.
        shape :
        deriv_filters : list of np.array() filters as [b_spline filter, 1st deriv filter, 2nd deriv filter]
        bcs : list of boundary conditions for the splines as strings ["wrap", "constant", ...]
        """

        self.coeffs_shape = coeffs_shape
        self.deriv_filters = np.array(deriv_filters, dtype=object)
        self.bcs = bcs

        self.dim = len(coeffs_shape) # geometric dimension
        self.deg_free = int(self.dim*(self.dim+1)/2) # degrees of freedom if symmetric
        self.hess_shape = (*self.coeffs_shape, self.deg_free)
        self.adj_deriv_filters = np.array([filt[::-1] for filt in self.deriv_filters], dtype=object) # do not simply to flip to allow for diff dimensions
        self.adj_bcs = adjoint_conv_bcs(self.bcs)
        self.d_combinations = deriv_combinations(self.dim, deriv_order = 2) # num of iterations produced should be eq to deg_free

        self.crop = tuple(self.dim*[slice(None)])
        self.slicing = self.deg_free*[tuple(self.dim*[slice(None)])]
                        # depending on the convolution and the length of the kernel
                        # [slice(*((int((max_len - len(filter))/2))*np.array([1,-1]))) for filter in self.deriv_filters]

        input_size = np.prod(coeffs_shape) # pycsou flattening
        output_size = self.deg_free*input_size # pycsou flattening
        super(bs_Hessian, self).__init__(shape=(output_size, input_size), dtype=dtype)

    def __call__(self, coeffs_flat: np.ndarray) -> np.ndarray:
        # Get Hessian from coefficients
        # should check padding make extra dimensions depending on bc? (here or before?)
        # maybe should add slicing?

        coeffs = coeffs_flat.reshape(self.coeffs_shape) # pycsou reshape
        hess = np.zeros(self.hess_shape)

        for i, combi in enumerate(self.d_combinations):
            hess[..., i] = convolve_coeffs(coeffs, self.deriv_filters[combi], self.bcs) # assume filters are separable in each dimension

        return hess.ravel() # pycsou flatten


    def adjoint(self, hess_flat: np.ndarray) -> np.ndarray:
        # not sure about the boundary conditions of the adjoint
        # if filters are symmetric should be (different dimensions appart) self-adjoint

        hess = hess_flat.reshape(self.hess_shape) # pycsou reshape
        coeffs = np.zeros(self.coeffs_shape) # 0th position indexes the deg_free of the matrix

        for i, combi in enumerate(self.d_combinations):
            coeffs[self.slicing[i]] += convolve_coeffs(hess[..., i], self.adj_deriv_filters[combi], self.adj_bcs)

        return coeffs[self.crop].ravel() # pycsou flatten


class Reshape(LinearOperator):
    # Pass from nD coefficients to 1D and vice-versa

    def __init__(self, shape_in, shape_out):

        assert (np.prod(shape_in) == np.prod(shape_out)), "Shapes do not match"
        self.shape_in = shape_in
        self.shape_out = shape_out

    def __call__(self, x: np.ndarray) -> np.ndarray:

        return np.reshape(x, self.shape_in)

    def adjoint(self, x: np.ndarray) -> np.ndarray:

        return np.reshape(x, self.shape_out)

    def inverse(self, x: np.ndarray) -> np.ndarray:

        return self.adjoint(x)

if __name__ == '__main__':
    from bs_utils import bs_deriv_filter, bs_to_filter, bs
    order = 3
    deriv_order = 2 # Hessian
    bs_filters = [bs_deriv_filter(i) for i in range(deriv_order+1)]
    bs_filters[0] = bs_to_filter(order, int_grid=True)
    bs_filters[1]= np.array([1., 0 ,-1.])
    coesp =(5,8)
    hesp = coesp + (3,)
    xyd = np.random.rand(np.prod(hesp)).reshape(hesp)
    xyd = (xyd**2 +3.)/100.
    xy = np.random.rand(np.prod(coesp)).reshape(coesp)
    xy = (xy**3 +5.)/1000.
    bh = bs_Hessian(coesp, bs_filters, ["wrap", "constant"])
    #bh(xy).reshape(hesp)
    ra = bh.adjoint(xyd).reshape(coesp)
    r = bh(xy).reshape(hesp)

    np.sum(ra*xy)-np.sum(xyd*r)
    bh.d_combinations
