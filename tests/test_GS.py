## Temporarily adding path
import sys
from pathlib import Path
sys.path.append(str(Path().absolute()))

## === Test Start ===
import numpy as np
from pyphaseretrieve  import algos
from pyphaseretrieve  import linop


class GS_algos_test(object):
    def __init__(self) -> None:
        print('GS test Start\n===============')

    def generate_rand2d_x(self, in_dim:int):
        x       = (np.random.randn(in_dim, in_dim) + 1j * np.random.randn(in_dim, in_dim)).astype(np.complex128) 
        return x

    def test_rand2d(self) -> None:
        # 1. Generate the groundtruth of image 
        in_dim  = 50
        x       = self.generate_rand2d_x(in_dim=in_dim)

        # 2. intensity image obtaining 
        near_y  = np.abs(x)**2              
        far_y   = np.abs(np.fft.fft2(x))**2 

        # 3. GS method solver
        GS_method        = algos.GerchbergSaxton(near_y,far_y)

        initial_est      = np.angle(x) + np.random.randn(in_dim,in_dim)*0.8  # choose initial guess close to target
        field_real_space = GS_method.iterate(initial_est= initial_est, n_iter= 2000)

        print("Result correlation:")
        _x                = np.ravel(x)
        field_real_space = np.ravel(field_real_space)
        print(np.abs( (field_real_space.T.conj() @ _x) /  (np.linalg.norm(field_real_space)*np.linalg.norm(_x)) ))

        print("Result correlation without optimize:")
        _initial_est = np.ravel(initial_est)
        print(np.abs( (_initial_est.T.conj() @ _x) /  (np.linalg.norm(_initial_est)*np.linalg.norm(_x)) ))


if __name__ == '__main__':
    GS_test = GS_algos_test()
    GS_test.test_rand2d()