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

        # 2. GD method solver
        loss_function = loss.loss_perturbative_based(n_iter= 50, lr= 1e-7)
        rand_GD_method = algos.GradientDescent(pr_model,loss_func= loss_function, line_search= None, acceleration=None)
        x_est = rand_GD_method.iterate(y=y,initial_est=None,n_iter=100,lr=-1)

        # 3. print final correlation
        print("Result correlation:")
        print(np.abs( (x_est.T.conj() @ x) /  (np.linalg.norm(x_est)*np.linalg.norm(x)) ))

if __name__ == '__main__':
    PPR_test = PPR_algos_test()

    PPR_test.test_rand1d_case()