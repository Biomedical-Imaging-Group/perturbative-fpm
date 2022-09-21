from __init__ import array_lib as np

def bs(eval, deg, positive=True):
    if positive:
        ax = eval
    else:
        ax = np.abs(eval)

    #h_sup = 0.5*(1+deg)
    if deg==0:
        return (ax<0.5)
    elif deg==1:
        return (ax<1.)*(1. - ax)
    elif deg==3:
        return (ax<2.)*(\
                         (ax<1)*(2./3. - ax**2 + 0.5*ax**3) \
                        +(ax>=1)*(1./6.)*((2. - ax)**3) \
                       )

    # Callables in piecewise not supported in cupy
    # elif deg==3:
    #     return np.piecewise(ax, \
    #                          [ax<1, ax>=1, ax>=2], \
    #                          [lambda x: 2./3. - x**2 + 0.5*x**3, \
    #                           lambda x: (1./6.)*((2. - x)**3), \
    #                           0])