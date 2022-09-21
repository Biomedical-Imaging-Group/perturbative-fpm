import math
import scipy.ndimage as nd
import scipy.signal as sg
import numpy as np


def bs(eval_at_points, order):
    return sg.bspline(eval_at_points, order) # bugs with integer points


def bs_to_filter(order, int_grid=True): # methods assume grid is regular so mayb no point
    half_domain = math.ceil((order+1)/2)
    if int_grid:
        half_domain -= 1
    return bs(np.arange(-half_domain, half_domain+1, dtype=float), order)


def bs_deriv_filter(deriv):
    pascal = [1.]
    for i in range(deriv):
        pascal.append(-1*pascal[i]*(deriv-i)/(i+1))
    return np.array(pascal)


def convolve_coeffs(coeffs, bs_filters, bcs): # successive separable convolutions
    bs_conv = coeffs
    for i, bc in enumerate(bcs):
        bs_conv = nd.convolve1d(bs_conv, bs_filters[i], mode=bc, axis=i) #oaconvolve #look for convolves that changes fft or mult according to speed
    return bs_conv


def bcs_to_value(bcs, period=1, ct=0):
    return np.array([bc_to_value(bc, period, ct) for bc in bcs])


def bc_to_value(bc, period=1, ct=0):
    if bc == "wrap":
        return period
    elif bc == "constant":
        return ct
    else:
        raise ValueError("This kind of boundary condition is not supported.") #ValueError or RuntimeError


def adjoint_conv_bcs(bcs):
    return [adjoint_conv_bc(bc) for bc in bcs]


def adjoint_conv_bc(bc):
    #if bc is not "wrap":
    #    bc = "constant"
    return bc


def eval_bspline_on_grid(coeffs, order, bcs):
    bs_filter = np.array(coeffs.ndim*[bs_to_filter(order)])
    print(bs_filter, convolve_coeffs(coeffs, bs_filter, bcs).shape)
    return convolve_coeffs(coeffs, bs_filter, bcs)


""" def eval_bspline(xys, coeffs, order):
    closest_grid_coords = np.rint(xys)
    n_evals = len(xys)
    half_support = support/2
    support_vector = np.arange(-half_support, half_support+1, dtype=int) # assuming the basis is centered at 0, half support each way. assume symmetric so can build Sijk with ::-1
    n_coeffs_involved = (len(support_vector))**dim
    closest_grid_coords = project_on_grid(sample_coords, grid=measuring_grid)
    dist_to_grid = sample_coords - closest_grid_coords """
