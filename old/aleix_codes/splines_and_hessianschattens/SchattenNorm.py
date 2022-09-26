from typing import Tuple, Optional, Union
from numbers import Number
import numpy as np
from pycsou.core.functional import ProximableFunctional

flat_index_2d = (np.s_[..., 0], np.s_[..., 1], np.s_[..., 1], np.s_[..., -1], np.s_[:-1])
erect_index_2d = (np.s_[...,0,0], np.s_[...,0,1], np.s_[...,1,0], np.s_[...,1,1], np.s_[:-2])
eps = 2.*np.finfo(float).eps


def erect_symmetric_matrix(flat_mat):
    # from (x,...,z, m12) to (x,...,z,m1,m2) when symmetric
    *shape, deg_free = flat_mat.shape
    dim = int(np.sqrt(deg_free*2))
    assert(dim*(dim+1)/2 == deg_free), "degrees of freedom do not match dimensions"

    row, col = np.triu_indices(dim)

    erect_mat = np.zeros((*shape, dim, dim), dtype=flat_mat.dtype)
    erect_mat[..., col, row] = np.conjugate(flat_mat) # first lower (order does not matter because diagonal is real)
    erect_mat[..., row, col] = flat_mat # then upper
    return erect_mat


def deflate_symmetric_matrix(erect_mat):
    # from (x,...,z, m1,m2) to (x,...,z, m12) when symmetric
    *shape, dim, dim = erect_mat.shape
    deg_free = int(dim*(dim+1)/2)

    row, col = np.triu_indices(dim)

    flat_mat = np.zeros((*shape, deg_free), dtype=erect_mat.dtype)
    flat_mat[..., :] = erect_mat[..., row, col]
    return flat_mat


def svd(mat_tensor, compute_basis = True, hermitian = False, flat = False, closed_formula = False):

    if closed_formula:
        return svd_2d(mat_tensor, compute_basis = compute_basis, flat=flat)

    else:
        if flat: mat_tensor = erect_symmetric_matrix(mat_tensor)
        return np.linalg.svd(mat_tensor, compute_uv = compute_basis, hermitian = hermitian)


def rebuild_svd(U, s, V, flat = False, closed_formula = False):

    if closed_formula:
        rebuilt = rebuild_svd_2d(U, s, V)
        return rebuilt if flat else erect_symmetric_matrix(rebuilt)

    else:
        rebuilt = (U[..., :, :]*s[..., np.newaxis, :]) @ V
        return rebuilt if not flat else deflate_symmetric_matrix(rebuilt)
        # (U*s) @ V; #U @ np.diag(s) @ V


def svd_2d(mat_tensor, compute_basis = True, flat = False):
    # closed formula for 2x2 matrices
    # changed the function to hermitian only
    ii, ij, ji, jj, sz = flat_index_2d if flat else erect_index_2d
    shape_out = (*mat_tensor.shape[sz], 2) # hermit output shape

    tr = mat_tensor[ii] + mat_tensor[jj]
    dt = (mat_tensor[ii] - mat_tensor[jj])**2 + 4*mat_tensor[ij]*mat_tensor[ji]
    # simplified from tr**2 - 4*( mat_tensor[ii]*mat_tensor[jj] - mat_tensor[ij]*mat_tensor[ji])
    singvals = np.zeros(shape_out)
    singvals[..., 0] = 0.5*(tr+np.sqrt(dt))
    singvals[..., 1] = 0.5*(tr-np.sqrt(dt))

    if compute_basis:
        singvec = np.zeros(shape_out)
        zero_ji = (np.abs(mat_tensor[ji])<eps).astype(int)
        singvec[..., 0] = (1-zero_ji)*(singvals[..., 0] - mat_tensor[jj])
        singvec[..., 1] = (1-zero_ji)*mat_tensor[ji] # from generic 2x2: e_1,y = e_2,y = ij
            # decided to focus on Hermitian only, so commenting this part away
            # if not hermitian: #eigenvecs not orthonormal
            #     zero_ij = (np.abs(mat_tensor(ij))<eps).astype(int)
            #     singvecs[..., 0] = singvec[..., 0] + zero_ji*(1-zero_ij)*mat_tensor[ij]
            #     singvecs[..., 1] = singvec[..., 1] + zero_ji*(1-zero_ij)*(singvals[..., 0] - mat_tensor[ii])
            #
            #     singvecs[..., 2] = (1-zero_ji)*(singvals[..., 1] - mat_tensor[jj]) + zero_ji*(1-zero_ij)*mat_tensor[ij]
            #     singvecs[..., 3] = (1-zero_ji)*mat_tensor[ji] + zero_ji*(1-zero_ij)*(singvals[..., 1] - mat_tensor[ii])
            #     return singvecs[..., :2], singvals, singvecs[..., 2:]

        norm = np.sqrt((singvals[..., 0] - mat_tensor[jj])**2 + mat_tensor[ji]**2) + zero_ji
        singvec[..., 0] = singvec[..., 0]/norm + zero_ji
        singvec[..., 1] /= norm
        return None, singvals, singvec
        # Hermitian only returns one singvec because the other can be computed from the first

    return singvals


def rebuild_svd_2d(U, s, V): # for 2x2 hermitian only, so now U is not required (passed as None)

    mat_tensor = np.zeros((*s.shape[:-1], 3))
    # if symmetric then eigenvecs are orthonormal so inverse of change of basis is just the transpose
    # also orthonormal in 2x2 means e_1,x = e_2,y, e_1,y = -e_2,x
    # https://people.math.harvard.edu/~knill/teaching/math21b2004/exhibits/2dmatrices/index.html
    mat_tensor[..., 0] = s[..., 0]*V[...,0]**2 + s[..., 1]*V[...,1]**2
    mat_tensor[..., 1] = (s[..., 0]-s[..., 1])*V[..., 0]*V[..., 1]
    mat_tensor[..., 2] = s[..., 0]*V[..., 1]**2 + s[..., 1]*V[...,0]**2

    return mat_tensor


def frobenius(x, flat):
    if flat:
        *shape, deg_free = x.shape
        dim = int(np.sqrt(deg_free*2))
        diag_inds = np.cumsum(np.arange(dim+1,1,-1))-dim-1
        x2 = x**2
        return np.sqrt(2*np.sum(x2, axis=-1) - np.sum(x2[..., diag_inds], axis=-1))
    else:
        return np.sqrt(np.einsum('...ijk,...ijk->...i', x, x))


def prox_l2(x, flat, tau):
    Frob = frobenius(x, flat)
    slc = np.s_[tuple([...]+(2-flat)*[np.newaxis])]
    with np.errstate(divide='ignore'):
        return np.maximum(1-tau/Frob, 0)[slc]*x


def proj_l2_ball(x, flat, radius):
    Frob = frobenius(x, flat)
    slc = np.s_[tuple([...]+(2-flat)*[np.newaxis])]
    if Frob <= radius:
        return x
    else:
        return radius * x / Frob[slc]


def init_proj_lq_ball(q):
    from pycsou.math.prox import proj_l1_ball, proj_linfty_ball #, proj_l2_ball
    assert (q in [np.inf, 1, 2]), "as of now, the p-norm proximal is only supported for q = 1, 2, inf" # in the Schatten context

    if q == np.inf:
        return proj_linfty_ball
    elif q == 1:
        return proj_l1_ball
    elif q == 2:
        return proj_l2_ball # do not take main proj_l2, adapt to Schatten


class LpSchattenNorm(ProximableFunctional):
    r"""
    Class for Lp Schatten Norm (i.e. the sum of the lp norm of the singular values
    of all matrices (one per point in the n-dim domain)).
    The proximity operator is computed by sv-decomposing, projecting the svs onto
    the conjugate ball and sv-recomposing with the original basis (except for p=2).
    """

    def __init__(self, hess_shape: Tuple[int, ...], p: Number = 1, hermitian: bool = False, flat: bool = False):
        r"""
        Parameters
        ----------
        shape: tuple of ints
            Input dimension
        p: int or float or np.infty
            Order of the norm
        hermitian: bool
        flat: bool
            Whether the matrices are flattened (only for hermitian)
        """

        self.hess_shape = hess_shape
        self.p = p
        with np.errstate(divide='ignore'): self.q = 1/(1-1/np.float64(p)) # 1/p+1/q=1  #q=p/(np.float64(p)-1) fails at np.infty
        self.proj_lq_ball = init_proj_lq_ball(self.q)
        self.flat = flat
        self.hermitian = hermitian if not flat else True
        self.closed_formula = (((flat and hess_shape[-1] == 3)
                                 or (not flat and hess_shape[-2:] == (2, 2)))
                               and hermitian) # only 2D hermitian as of now

        input_size = np.prod(hess_shape) # pycsou flattening
        # output_size = 1 # pycsou flattening
        super(LpSchattenNorm, self).__init__(dim=input_size, data=None, is_differentiable=False, is_linear=False) # is_convex = True # pycsou dim


    def __call__(self, hess_flat: Union[Number, np.ndarray]) -> Number:
        # hess is array with last one/two (flat/not-flat) dimensions being the matrix
        hess = hess_flat.reshape(self.hess_shape) # pycsou reshape

        if not self.p==2:
            singvals = svd(hess, compute_basis = False, hermitian = self.hermitian,
                                 flat = self.flat, closed_formula = self.closed_formula)
            s_norm = np.sum(np.linalg.norm(singvals, ord=self.p, axis=-1), axis=None)

        else: # no need for svd because the norm is Frobenius'
            s_norm = np.sum(frobenius(hess, self.flat), axis=None)

        return s_norm # pycsou flatten


    def prox(self, x_flat: Union[Number, np.ndarray], tau: Number) -> Union[Number, np.ndarray]:
        # Proximal projects the svals on the lq ball and reconstructs with the svecs
        x = x_flat.reshape(self.hess_shape) # pycsou reshape

        if not self.p==2:
            Uvecs, singvals, Vvecs = svd(x, compute_basis = True, hermitian = self.hermitian,
                                            flat = self.flat, closed_formula = self.closed_formula)

            singvals -= tau*self.proj_lq_ball(singvals/tau, radius=1) #(singvals/tau, radius=1)
            prox_x = rebuild_svd(Uvecs, singvals, Vvecs,
                                            flat = self.flat, closed_formula = self.closed_formula)

        else: # no need for svd because the norm is Frobenius'
            prox_x = x-tau*self.proj_lq_ball(x/tau, self.flat, radius=1)

        return prox_x.ravel() # pycsou flatten

    def project(self, x_flat: Union[Number, np.ndarray], radius: Number) -> Union[Number, np.ndarray]:
        # projection only. should not repeat code but not calling project in prox to save some subtractions.
        x = x_flat.reshape(self.hess_shape) # pycsou reshape

        if not self.p==2:
            Uvecs, singvals, Vvecs = svd(x, compute_basis = True, hermitian = self.hermitian,
                                            flat = self.flat, closed_formula = self.closed_formula)

            singvals = self.proj_lq_ball(singvals, radius=radius) #(singvals/tau, radius=1)
            project_x = rebuild_svd(Uvecs, singvals, Vvecs,
                                            flat = self.flat, closed_formula = self.closed_formula)

        else: # no need for svd because the norm is Frobenius'
            project_x = self.proj_lq_ball(x, self.flat, radius=radius)

        return project_x.ravel() # pycsou flatten


if __name__ == '__main__':

    # Testing slicing bc_behavior
    slc= np.s_[..., 0]
    np.array([[  0,   1,   2], [  3,   4,   5],[  6,   7,   8]])[slc]

    ## Testing
    hs_shape = (4,6,3)
    np.array([1,0,1])

    lsn = LpSchattenNorm(hs_shape, p=1, hermitian=True, flat=True)
    lsn2 = LpSchattenNorm(hs_shape, p=2, hermitian=True, flat=True)
    lsnf = LpSchattenNorm(hs_shape, p=np.infty, hermitian=True, flat=True)

    hess_id = np.array(4*6*[1,0,1]).reshape(hs_shape) #1,1
    lsn(hess_id)/2
    lsn2(hess_id)/np.sqrt(2)
    lsnf(hess_id)
    hess_id = 2.*np.array(4*6*[1,0,1]).reshape(hs_shape) #1,1
    lsn(hess_id)/2
    lsn2(hess_id)/np.sqrt(2)
    lsnf(hess_id)
    other = np.array(4*6*[2,3,2]).reshape(hs_shape) #5,-1
    lsn(other)/(5+1)
    lsn2(other)/np.sqrt(5**2+1)
    lsnf(other)/5
    other = np.array(4*6*[8,8,8]).reshape(hs_shape) #16,0
    lsn(other)/16
    lsn2(other)/16
    lsnf(other)/16
    other = np.array(4*3*[8,8,8]+4*3*[2,3,2]).reshape(hs_shape) #16,0
    lsn(other)/((5+1+16)/2)
    lsn2(other)/((16+np.sqrt(5**2+1))/2)
    lsnf(other)/((16+5)/2)


    ## Testing cases
    # flat and closed
    #import timeit #timeit.timeit("
    import time
    leeway = 30.
    x, y = 2*[2] # at 10000 the timings are 20s,41s,104s,110s
    m = 3
    a1 = np.arange(x*y*m).reshape(x,y,m) # err grows with matrix size

    # np.linalg.norm(a2, ord=2, axis=(-2,-1)) # not expected behavior must code explicitly

    # not-flat and closed
    start = time.time()
    u,s,v = svd_2d(a1, compute_basis = True, flat = True)
    err = rebuild_svd_2d(u,s,v)-a1
    np.max(err), np.mean(np.abs(err))
    np.max(err)<leeway*eps
    time.time()-start
    n = np.linalg.norm(s, ord=1, axis=-1)
    n, np.sum(n,axis=None)

    # not-flat and closed
    start = time.time()
    a1e = erect_symmetric_matrix(a1)
    u,s,v = svd(a1e, compute_basis = True, hermitian = True, flat = False, closed_formula = True)
    err = a1e-rebuild_svd(u, s, v, flat = False, closed_formula = True)
    np.max(err), np.mean(np.abs(err))
    np.max(err)<leeway*eps
    time.time()-start
    n = np.linalg.norm(s, ord=1, axis=-1)
    n, np.sum(n,axis=None)

    # flat and not-closed
    start = time.time()
    u,s,v = svd(a1, compute_basis = True, hermitian = True, flat = True, closed_formula = False)
    err = a1-rebuild_svd(u, s, v, flat = True, closed_formula = False)
    np.max(err), np.mean(np.abs(err))
    np.max(err)<leeway*eps
    time.time()-start
    n = np.linalg.norm(s, ord=1, axis=-1)
    n, np.sum(n,axis=None)

    # not-flat and not-closed
    start = time.time()
    u,s,v = svd(a1e, compute_basis = True, hermitian = True, flat = False, closed_formula = False)
    err = a1e-rebuild_svd(u, s, v, flat = False, closed_formula = False)
    np.max(err), np.mean(np.abs(err))
    np.max(err)<leeway*eps
    time.time()-start
    n = np.linalg.norm(s, ord=1, axis=-1)
    n, np.sum(n,axis=None)
