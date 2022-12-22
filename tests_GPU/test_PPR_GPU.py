## Temporarily adding path
import sys
from pathlib import Path
sys.path.append(str(Path().absolute()))

## === Test Start ===
import numpy as np
import cupy as cp
from pyphaseretrieve        import linop
from pyphaseretrieve        import phaseretrieval
from pyphaseretrieve        import algos
from pyphaseretrieve        import loss

class PPR_algos_test(object):   
    def __init__(self) -> None:
        print('PPR test Start\n===============')

    def generate_rand1d_x(self, in_dim):
        x = (cp.random.randn(in_dim, ) + 1j * cp.random.randn(in_dim,))
        return x

    def test_rand1d_case(self) -> None:
        print('Test: Random matrix case')
        # 1. LinOp Build
        in_dim  = 800
        out_dim = 16000
        op_matrix = linop.LinOpMatrix(cp.random.randn(out_dim, in_dim) + 1j * cp.random.randn(out_dim, in_dim))
        pr_model = phaseretrieval.PhaseRetrievalBase(linop= op_matrix)
        
        x = self.generate_rand1d_x(in_dim)
        y = np.abs(pr_model.apply(x))**2
        print('model: in_dim: {}, out_dim: {}, oversampling ratio: {}'.format(in_dim, out_dim, out_dim/in_dim))

        # 2.  solver
        ppr_model = algos.PerturbativePhase(pr_model)
        initial_est = cp.ones(shape= (in_dim,), dtype= np.complex128)
        # x_est = ppr_model.iterate_GradientDescent(y= y, initial_est= initial_est, n_iter= 15, linear_n_iter= 50, lr=1e-9)
        x_est = ppr_model.iterate_ConjugateGradientDescent(y= y, initial_est= initial_est,n_iter=15, linear_n_iter=15)

        # 3. print final correlation
        print("Result correlation:")
        print(np.abs( (x_est.T.conj() @ x) /  (np.linalg.norm(x_est)*np.linalg.norm(x)) ))
    
    def test_Ptychography1d_case_without_spectral(self):
        ## 1D ptychography
        print('Test: 1D ptycho case WITHOUT spectral method')
        # 1. new X
        in_dim = 800
        x      = self.generate_rand1d_x(in_dim)

        # 2.Probe create
        ptycho_radius = 200
        sampling_grid = np.abs(cp.linspace(-int(np.ceil(in_dim/2)), in_dim//2, in_dim))
        probe = (cp.ones(in_dim,).astype(cp.complex128) * (sampling_grid < ptycho_radius)) 

        # 3. Ptychogram create
        pr_model = phaseretrieval.Ptychography1d(probe,n_img=10)
        print(f"The overlap is {pr_model.overlap_rate()}")

        y = np.abs(pr_model.apply(x))**2

        # 4. solver
        ppr_model = algos.PerturbativePhase(pr_model)
        initial_est = cp.ones(shape= (in_dim,), dtype= cp.complex128)
        # x_est = ppr_model.iterate_GradientDescent(y= y, initial_est= initial_est, n_iter= 100, linear_n_iter= 40, lr=2e-4)
        x_est = ppr_model.iterate_ConjugateGradientDescent(y= y,initial_est= initial_est,n_iter=25, linear_n_iter=15)

        # 5. print final correlation
        print("Result correlation:")
        print(np.abs( (x_est.T.conj() @ x) /  (np.linalg.norm(x_est)*np.linalg.norm(x))))

    def test_Ptychography1d_case_with_spectral(self):
        ## 1D ptychography
        print('Test: 1D ptycho case WITH spectral method')
        # 1. new X
        in_dim = 800
        x      = self.generate_rand1d_x(in_dim)

        # 2.Probe create
        ptycho_radius = 250
        sampling_grid = np.abs(cp.linspace(-in_dim//2, int(np.ceil(in_dim/2))-1, in_dim))
        probe = cp.ones(in_dim,).astype(cp.complex128) * (sampling_grid < ptycho_radius)

        # 3. Ptychogram create
        pr_model = phaseretrieval.Ptychography1d(probe,n_img=10)
        print(f"The overlap is {pr_model.overlap_rate()}")

        y = np.abs(pr_model.apply(x))**2

        # 4. solver
        ppr_model = algos.PerturbativePhase(pr_model)
        Spec_method = algos.SpectralMethod(pr_model= pr_model)

        initial_est = cp.random.randn(in_dim)
        x_spec = Spec_method.iterate(y= y, initial_est= initial_est)
        x_est =ppr_model.iterate_ConjugateGradientDescent(y= y, initial_est= x_spec, n_iter=25, linear_n_iter=20)

        # 5. print final correlation
        print("Result correlation:")
        print(np.abs( (x_est.T.conj() @ x) /  (np.linalg.norm(x_est)*np.linalg.norm(x))))

if __name__ == '__main__':
    PPR_test = PPR_algos_test()

    # PPR_test.test_rand1d_case()
    # print('---------------')
    PPR_test.test_Ptychography1d_case_without_spectral()
    print('---------------')
    PPR_test.test_Ptychography1d_case_with_spectral()