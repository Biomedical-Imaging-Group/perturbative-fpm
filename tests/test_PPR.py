## Temporarily adding path
import sys
from pathlib import Path
sys.path.append(str(Path().absolute()))

## === Test Start ===
import numpy as np
from pyphaseretrieve        import linop
from pyphaseretrieve        import phaseretrieval
from pyphaseretrieve        import algos
from pyphaseretrieve        import loss

class PPR_algos_test(object):   
    def __init__(self) -> None:
        print('PPR test Start\n===============')

    def generate_rand1d_x(self, in_dim):
        x = (np.random.randn(in_dim, ) + 1j * np.random.randn(in_dim,))
        return x

    def test_rand1d_case(self) -> None:
        print('Test: Random matrix case')
        # 1. LinOp Build
        in_dim  = 100
        out_dim = 800
        op_matrix = linop.LinOpMatrix(np.random.randn(out_dim, in_dim) + 1j * np.random.randn(out_dim, in_dim))
        pr_model = phaseretrieval.PhaseRetrievalBase(linop= op_matrix)
        
        x = self.generate_rand1d_x(in_dim)
        y = np.abs(pr_model.apply(x))**2
        print('model: in_dim: {}, out_dim: {}, oversampling ratio: {}'.format(in_dim, out_dim, out_dim/in_dim))

        # 2.  solver
        ppr_model = algos.PerturbativePhase(pr_model)
        x_est = ppr_model.iterate_CGD(y= y,n_iter=15, CGD_n_iter=15)
        # x_est = ppr_model.iterate_GD(y= y, n_iter= 15, GD_n_iter= 25, lr=1e-7)

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
        sampling_grid = np.abs(np.linspace(-int(np.ceil(in_dim/2)), in_dim//2, in_dim))
        probe = (np.ones(in_dim,).astype(np.complex128) * (sampling_grid < ptycho_radius)) 

        # 3. Ptychogram create
        pr_model = phaseretrieval.Ptychography1d(probe,n_img=10)
        print(f"The overlap is {pr_model.overlap_rate()}")

        y = np.abs(pr_model.apply(x))**2

        # 4. solver
        ppr_model = algos.PerturbativePhase(pr_model)
        x_est = ppr_model.iterate_CGD(y= y,n_iter=25, CGD_n_iter=15)

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
        sampling_grid = np.abs(np.linspace(-in_dim//2, int(np.ceil(in_dim/2))-1, in_dim))
        probe = np.ones(in_dim,).astype(np.complex128) * (sampling_grid < ptycho_radius)

        # 3. Ptychogram create
        pr_model = phaseretrieval.Ptychography1d(probe,n_img=10)
        print(f"The overlap is {pr_model.overlap_rate()}")

        y = np.abs(pr_model.apply(x))**2

        # 4. solver
        ppr_model = algos.PerturbativePhase(pr_model)
        Spec_method = algos.SpectralMethod(pr_model= pr_model)

        x_spec = Spec_method.iterate(y= y)
        x_est =ppr_model.iterate_CGD(y= y,n_iter=25, CGD_n_iter=20)

        # 5. print final correlation
        print("Result correlation:")
        print(np.abs( (x_est.T.conj() @ x) /  (np.linalg.norm(x_est)*np.linalg.norm(x))))

if __name__ == '__main__':
    PPR_test = PPR_algos_test()

    PPR_test.test_rand1d_case()
    # print('---------------')
    # PPR_test.test_Ptychography1d_case_without_spectral()
    # print('---------------')
    # PPR_test.test_Ptychography1d_case_with_spectral()