from __init__ import array_lib as np
from plt import plt


class sum_differentiable():
    def __init__(self, A, B):
        self.A=A
        self.B=B
        self.lip = A.lip+B.lip
        self.eigen_min = A.eigen_min+B.eigen_min #only in certain cases
        self.dim = A.dim

    def __call__(self, x):
        return self.A(x) + self.B(x)
    def jacobianT(self, x):
        return self.A.jacobianT(x) + self.B.jacobianT(x)


class squared_l2_norm(): #DifferentiableFunctional
    # 1/2 * alpha * ||x||^2
    def __init__(self, dim_in, alpha=2.):
        self.alpha = alpha
        self.lip = alpha
        self.eigen_min = alpha
        self.dim = [dim_in, 1]

    def __call__(self, x):
        return 0.5*self.alpha*np.linalg.norm(x)**2

    def jacobianT(self, x): 
        return self.alpha*x


class weighted_squared_l2_loss(): #DifferentiableLoss/DifferentiableFunctional
    # 1/2 * ||Px-b||^2_A
    def __init__(self, P, A=None, data=None, PTAP=None, PTAdata=None):
        self.P = P
        self.A = A if A is not None else lambda x: x
        self.data = data if data is not None else np.zeros_like(P.dim[0])
        self.lip = P.lip*A.lip
        self.eigen_min = P.eigen_min*A.eigen_min
        self.dim = [P.dim[0], 1]
        
        self.PTAP = PTAP
        if self.PTAP is not None and PTAdata is None:
            self.PTAdata = self.P.adjoint(self.A(data))
        else:
            self.PTAdata = PTAdata

    def __call__(self, x):
        #return np.linalg.norm(x)**2 #.ravel()
        err = self.P(x)-self.data
        return 0.5*np.vdot(err, self.A(err))

    def jacobianT(self, x): 
        if self.PTAP is None:
            err = self.P(x)-self.data
            return self.P.adjoint(self.A(err))
        else:
            return self.PTAP(x)-self.PTAdata


class l12_norm():
    def __init__(self, shp):
        self.shp = shp
        self.size = int(np.prod(np.asarray(shp))) #np.prod(shp)
        self.ndim = len(shp)
    
    def __call__(self, x):
        return np.sum(self.l2(x))
        
    def project(self, x):    
        n = self.l2(x).clip(1.) #np.maximum(self.l2(x),1.)
        return (x.reshape(self.ndim,-1)/n).ravel()
    
    def l2(self,x):
        return np.linalg.norm(x.reshape(self.ndim,-1), axis=0)


class TV_cost():

    def __init__(self, shp, n_prox_iter, lam=1., proj=None, iter_func=None, acc='BT'):
        self.shp = shp
        self.size = int(np.prod(np.asarray(shp))) #np.prod(shp)
        self.ndim = len(shp)
        self.lam = lam #reg constant
        self.n_prox_iter = n_prox_iter

        from linear_operators import LinOpGrad
        self.Grad = LinOpGrad(shp) #should reshape like .reshape(ndim,-1), but when? proj should take this into account.
        self.norm = l12_norm(shp)
        self.iter_func = lambda x: x if iter_func is None else iter_func
        self.acc = acc

        self.tv_ct = 8.

        if proj is None:
            self.proj = lambda x: x
        elif isinstance(proj, (list, np.ndarray)): #interpret as a minmax constrain
            self.proj = lambda x: x.clip(*proj)
        elif callable(proj): #hasattr(proj, '__call__'):
            self.proj = proj
        else:
            raise Exception('wrong type')
        #self.proj_norm to add to call eventually


    def __call__(self, x):
        return self.lam*self.norm(self.Grad(x))
                               #l2 isotropic, l1 for anisotropic (no need to reshape for l1 because independent)
    def prox(self, y, tau):
        tau = self.lam*tau

        ilip = 1./(self.tv_ct*tau)
        x = np.zeros(self.size*self.ndim)
        x_prev = np.zeros_like(x)
        t_prev = 1.

        for i in range(self.n_prox_iter):
            x_temp = x + ilip*self.Grad(\
                               self.proj( \
                                   y-tau*self.Grad.adjoint(x)
                                ) )
            
            x_temp = self.norm.project(x_temp)

            if self.acc=="BT":
                t = 0.5 * (1 + np.sqrt(1 + 4 * t_prev ** 2))
            else:
                t = 1
            x = x_temp + (x_temp - x_prev) * (t_prev - 1) / t

            x_prev = x_temp
            t_prev = t
            self.n_prox_iter = self.iter_func(self.n_prox_iter)

        return self.proj(y-tau*self.Grad.adjoint(x_temp)) 


def powerit_PTAP(P, A, niter=10, stop=1.):
    # import scipy.sparse.linalg as spls
    # spls.eigh, spls.eig, spls.svd # LM and SM for largest and smallest value
    # non-negative

    x_prev = np.ones(P.dim[0])
    for i in range(niter):
        x = P.adjoint(A(P(x_prev)))

        r = x/x_prev
        lb = np.min(r)
        ub = np.max(r)
        x_prev = 2*x/(lb + ub)
        
        if ub/lb < stop: break
    print("number of iterations:", i)
    return ub