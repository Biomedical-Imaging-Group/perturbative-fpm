import numpy as np
import unittest
import pyphaseretrieve.linop as linop

class TestLinOpMatrix(unittest.TestCase):

    def setUp(self):
        in_dim = np.random.randint(low=1, high=10)
        out_dim = np.random.randint(low=1, high=10)
        self.M = linop.LinOpMatrix(np.random.randn(out_dim, in_dim))
    
    def test_dimensions_and_apply(self):    
        self.assertEqual(self.M.apply(np.random.randn(self.M.in_size)).size, 
                         self.M.out_size,
                         "Apply method inconsistent with output size")
        self.assertEqual(self.M.applyAdjoint(np.random.randn(self.M.out_size)).size, 
                         self.M.in_size,
                         "Apply Adjoint method inconsistent with input size")

    def test_adjoint_property(self):        
        in_vec = np.random.randn(self.M.in_size)
        out_vec = np.random.randn(self.M.out_size)
        self.assertTrue(np.allclose(
                        np.inner(self.M.apply(in_vec), out_vec),
                        np.inner(in_vec, self.M.applyAdjoint(out_vec))),
                         "Adjoint property not fulfilled")

    def test_rectangular_matrix_sizes(self):
        M = linop.LinOpMatrix(np.random.randn(10,5))
        self.assertEqual(10, M.out_size,
        msg="Output size of LinOpMatrix inconsistent with matrix")
        self.assertEqual(5, M.in_size,
        msg="Input size of LinOpMatrix inconsistent with matrix")

if __name__ == '__main__':
    unitest.main()