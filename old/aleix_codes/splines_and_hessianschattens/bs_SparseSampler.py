from pycsou.core.linop import LinearOperator
import numpy as np
import sparse as sp
import functools
#https://stackoverflow.com/questions/1208118/using-numpy-to-build-an-array-of-all-combinations-of-two-arrays

def project_on_grid(points, grid=None):
    return np.rint(points)


def outter_product_indexing(base_vectors): # one vector per point and per dimension
    dim = len(base_vectors)
    ein_args = (2*dim+1)*[None]
    outter_map = np.arange(dim+1)[::-1]
    if dim>1: outter_map[-1] = 1; outter_map[-2] = 0;
    ein_args[:-1:2] = base_vectors # not a copy, only a pointer
    ein_args[1::2] = np.stack((np.arange(dim),
                          dim*np.ones(dim, dtype=int)), # naming dimensions in base_vectors
                          axis=-1)
    ein_args[-1] = outter_map # output form to create outter product

    return ein_args
    # einsum indexing: in 3D it's [0,3], [1,3], [2,3] -> [0,1,2,3], equivalent to 'il, jl, kl->ijkl',
    # to add transpose simply reverse order of axes [0,1,2,3][::-1] or lkji
    # we do 'im, jm, km, lm -> mlkij',


def sparse_constructor(ijks, Vijk, shape):
    return sp.COO(ijks, Vijk, shape=shape)


class bs_SparseSampler(LinearOperator):
    # check what to do with the negatives? maybe displace data?

    def __init__(self, coeffs_shape, sample_coords, measuring_grid, basis, support, bc_behavior, dtype: type = np.float64):
        # Prepare sparse matrix SS that goes from coefficients to an array of the recovered function evaluated at the same place as the samples, ideally we look for SS*coeffs = data
        r"""
        Parameters
        ----------
        size : tuple of ints representing the shape of the input array.
        coeffs_shape : shape of the coefficients
        deriv_filters : list of np.array() filters as [b_spline filter, 1st deriv filter, 2nd deriv filter]
        bcs : list of behavior of the bcs in each axis in relation to the coefficients
        """

        # assuming samples as [[x,y],..., [x,y]] and assuming uniform grid for the basis
        self.coeffs_shape = coeffs_shape
        dim = sample_coords.shape[-1]
        n_samples = len(sample_coords)
        half_support = support/2
        support_vector = np.arange(-half_support, half_support+1, dtype=int) # assuming the basis is centered at 0, half support each way. assume symmetric so can build Sijk with ::-1
        n_coeffs_involved = (len(support_vector))**dim # (support+1)**dim # gross estimate (adding dim unneeded but sparse constructor will take care of them)

        closest_grid_coords = project_on_grid(sample_coords, grid=measuring_grid)
        dist_to_grid = sample_coords - closest_grid_coords
        nd_support_combinations = np.array(np.meshgrid(*dim*[support_vector])).T.reshape(-1,dim)
        assert(len(nd_support_combinations[...,0]) == n_coeffs_involved)

        # indexing the output
        ii = np.broadcast_to(np.arange(n_samples), (n_coeffs_involved, n_samples)).T.ravel()

        # indexing the input coefficients
        jjkk = (np.repeat(closest_grid_coords.astype(int).T, n_coeffs_involved, axis=1) + np.tile(nd_support_combinations.T, n_samples)) % bc_behavior.astype(int)[:,np.newaxis]

        # Lets do the minimum (dim*samples*support). Since basis not linear, cannot evaluate before summing. #added [::-1]
        evaluated_support_vector = basis((dist_to_grid[:, np.newaxis]+support_vector[::-1, np.newaxis]).T) # create uniform vectors around samples in all directions according to distance and evaluate
        Sijk = np.einsum(*outter_product_indexing(evaluated_support_vector)).ravel() # outter product by columns gives tensor of evaluations from vector


        self.SS = sparse_constructor((ii, *jjkk), Sijk, (n_samples,)+self.coeffs_shape)
        self.SS_adj = self.SS.transpose()
        ## ii is a vector of indices, jjkk has one such vector per dimension of the coeffs func

        input_size = np.prod(coeffs_shape) # pycsou flattening
        output_size = len(sample_coords) # pycsou flattening
        super(bs_SparseSampler, self).__init__(shape=(output_size, input_size), dtype=dtype, is_dense=False) #is_sparse=True (only when scipy sparse!)


    def __call__(self, coeffs_flat: np.ndarray) -> np.ndarray:
        # input coefficients get the values (1d vector) at the desired points

        coeffs = coeffs_flat.reshape(self.coeffs_shape) # pycsou reshape
        return sp.tensordot(self.SS, coeffs, axes=coeffs.ndim) # already flattened (pycsou)
        # self.SS*coeffs or np.dot(SS, coeffs) or SS._matvec(coeffs)


    def adjoint(self, evals: np.ndarray) -> np.ndarray:

        adj_coeffs = sp.tensordot(self.SS_adj, evals, axes=1).T # already in shape (pycsou)
        return adj_coeffs.ravel() # pycsou flatten
        # self.SS_adj*evals or np.dot(SS.T, evals) or SS.H or SS._rmatvec(evals)


    def HTH(self, coeffs_flat: np.ndarray) -> np.ndarray:

        return self.adjoint(self.__call__(coeffs_flat))


    def sp_HTH(self, coeffs_flat: np.ndarray) -> np.ndarray:

        coeffs = coeffs_flat.reshape(self.coeffs_shape) # pycsou reshape
        adj_coeffs = sp.tensordot(self.SSTSS, coeffs, axes=coeffs.ndim).T
        return adj_coeffs.ravel() # pycsou flatten


    #@functools.cached_property
    @property
    @functools.lru_cache()
    def SSTSS(self):
        return sp.tensordot(self.SS_adj, self.SS, axes=[-1,0])

    # pycsou specific: reshape when entering, flatten when exiting (make sure to pass the proper shape to inheritance)


if __name__ == '__main__':
    import bs_utils as bu

    domain_offset = np.array([10.,10.]) # this 4 to change dimension
    domain_length = np.array([80.,80.])
    coeffs_shape = (100, 102)
    bcs = ["constant", "wrap"]

    n_samples = 500
    order = 3

    domain_dim = len(domain_length)
    sample_coords = domain_offset + domain_length*np.random.rand(n_samples, domain_dim)
    period = domain_offset[0]+domain_length[0]

    bcs_behaviour = bu.bcs_to_value(bcs, period, 2*np.max(np.array(coeffs_shape)))
    bss = bs_SparseSampler(coeffs_shape, sample_coords, None, lambda args: bu.bs(args, order), order+1, bcs_behaviour)

    cos = np.zeros(coeffs_shape); cos[50,50]=1.
    cos = np.random.rand(np.prod(coeffs_shape)).reshape(coeffs_shape)
    evs = np.random.rand(n_samples)

    np.sum(bss.adjoint(evs)>0.2)
    n_samples*np.prod(coeffs_shape)
    bss.SS.nnz
    bss.SS_adj.nnz
    bss.SSTSS.nnz
    np.max(bss.adjoint(bss(cos))-bss.HTH(cos))
    np.max(bss.adjoint(bss(cos))-bss.sp_HTH(cos))

    bss.SS.density
    bss.SS_adj.density
    bss.SSTSS.density
