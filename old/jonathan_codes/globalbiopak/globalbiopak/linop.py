import torch
import numpy as np

## Base class

class BaseLinOp:
    def __init__(self):
        pass

    def __add__(self, other):
        if isinstance(other, BaseLinOp):
            return LinOpSum(self, other)
        else:
            return LinOpScalarSum(self, other)

    def __radd__(self, other):
        if isinstance(other, BaseLinOp):
            return LinOpSum(self, other)
        else:
            return LinOpScalarSum(self, other)
    
    def __sub__(self, other):
        if isinstance(other, BaseLinOp):
            return LinOpDiff(self, other)
        else:
            return LinOpScalarDiff(self, other)
    
    def __rsub__(self, other):
        if isinstance(other, BaseLinOp):
            return LinOpDiff(self, other)
        else:
            return LinOpScalarDiff(self, other)
        
    def __mul__(self, other):
        if isinstance(other, BaseLinOp):
            raise NameError('This case of multiplication between two LinOp objects has not been implemented yet. Please have a look at LinOpMul and contact us if you need additional functionalities.')
        else:
            return LinOpScalarMul(self, other)
        
    def __rmul__(self, other):
        if isinstance(other, BaseLinOp):
            raise NameError('This case of multiplication between two LinOp objects has not been implemented yet. Please have a look at LinOpMul and contact us if you need additional functionalities.')
        else:
            return LinOpScalarMul(self, other)
    
    def __matmul__(self, other):
        if isinstance(other, BaseLinOp):
            return LinOpComposition(self, other)
        else:
            raise NameError('The matrix multiplication operator can only be performed between two LinOp objects.')

## Default classes
    
class LinOpMatrix(BaseLinOp):
    def __init__(self, matrix):
        self.H = matrix
        self.in_size = matrix.shape[1]
        self.out_size = matrix.shape[0]
    
    def apply(self, x):
        return self.H @ x

    def applyAdjoint(self, x):
        return self.H.T.conj() @ x

class LinOpMul(BaseLinOp):
    """coefs is a vector for element-wise multiplication"""
    def __init__(self, coefs):
        self.coefs = coefs
        self.in_size = coefs.shape[0]
        self.out_size = coefs.shape[0]

    def apply(self, x):
        return self.coefs * x

    def applyAdjoint(self, x):
        return self.coefs.conj() * x
    
class LinOpConv(BaseLinOp):
    """Convolution operator, computed using FFT"""
    def __init__(self, Ffilter, epsilon=1e-5):
        self.Ffilter = Ffilter  # Filter in the Fourier domain
        self.epsilon = epsilon  # regularization constant for deconvolution
        self.in_size = Ffilter.shape[0]
        self.out_size = Ffilter.shape[0]

    def apply(self, x):
        return torch.fft.ifft2(self.Ffilter * torch.fft.fft2(x))

    def applyAdjoint(self, x):
        return torch.fft.fft2(1 / (self.Ffilter+self.epsilon) * torch.fft.ifft2(x))

class LinOpFFT(BaseLinOp):
    def __init__(self):
        self.in_size = -1
        self.out_size = -1

    def apply(self, x):
        return torch.fft.fft(x, norm="ortho")

    def applyAdjoint(self, x):
        return torch.fft.ifft(x, norm="ortho")

class LinOpIFFT(BaseLinOp):
    def __init__(self):
        self.in_size = -1
        self.out_size = -1

    def apply(self, x):
        return torch.fft.ifft(x, norm="ortho")

    def applyAdjoint(self, x):
        return torch.fft.fft(x, norm="ortho")

class LinOpFFT2(BaseLinOp):
    def __init__(self):
        self.in_size = -1
        self.out_size = -1

    def apply(self, x):
        return torch.fft.fft2(x, norm="ortho")

    def applyAdjoint(self, x):
        return torch.fft.ifft2(x, norm="ortho")

class LinOpIFFT2(BaseLinOp):
    def __init__(self):
        self.in_size = -1
        self.out_size = -1

    def apply(self, x):
        return torch.fft.ifft2(x, norm="ortho")

    def applyAdjoint(self, x):
        return torch.fft.fft2(x, norm="ortho")
    
class LinOpId(BaseLinOp):
    def __init__(self):
        self.in_size = -1
        self.out_size = -1
        
    def apply(self, x):
        return x
    
    def applyAdjoint(self, x):
        return x
    
class LinOpConstant(BaseLinOp):
    """WARNING: THIS IS NOT A LINOP, TO BE DISCUSSED"""
    def __init__(self, value=1):
        self.value = value
        self.in_size = -1
        self.out_size = -1
        
    def apply(self, x):
        return self.value
    
    def applyAdjoint(self, x):
        return 0
    
class LinOpFlip(BaseLinOp):
    def __init__(self):
        self.in_size = -1
        self.out_size = -1

    def apply(self, x):
        return torch.flip(x, dims=[i for i in range(x.dim())])

    def applyAdjoint(self, x):
        return torch.flip(x, dims=[i for i in range(x.dim())])

class LinOpRoll(BaseLinOp):
    def __init__(self, shifts, dims):
        self.in_size = -1
        self.out_size = -1
        self.shifts = shifts
        self.dims = dims

    def apply(self, x):
        return torch.roll(x, shifts=self.shifts, dims=self.dims)

    def applyAdjoint(self, x):
        return torch.roll(x, shifts=[-shift for shift in self.shifts], dims=self.dims)

    
## Utils classes
    
class LinOpComposition(BaseLinOp):
    def __init__(self, LinOp1, LinOp2):
        self.LinOp1 = LinOp1
        self.LinOp2 = LinOp2
        self.in_size = LinOp1.in_size if LinOp1.in_size != -1 else LinOp2.in_size
        self.out_size = LinOp2.out_size if LinOp2.out_size != -1 else LinOp1.out_size

    def apply(self, x):
        return self.LinOp1.apply(self.LinOp2.apply(x))

    def applyAdjoint(self, x):
        return self.LinOp2.applyAdjoint(self.LinOp1.applyAdjoint(x))

class LinOpSum(BaseLinOp):
    def __init__(self, LinOp1, LinOp2):
        self.LinOp1 = LinOp1
        self.LinOp2 = LinOp2
        self.in_size = np.maximum(LinOp1.in_size, LinOp2.in_size)  # it is -1 if size is undefined
        self.out_size = np.maximum(LinOp1.out_size, LinOp2.out_size)

    def apply(self, x):
        return self.LinOp1.apply(x) + self.LinOp2.apply(x)

    def applyAdjoint(self, x):
        return self.LinOp2.applyAdjoint(x) + self.LinOp1.applyAdjoint(x)
    
class LinOpScalarSum(BaseLinOp):
    def __init__(self, LinOp, other):
        self.LinOp = LinOp
        self.scalar = other
        self.in_size = LinOp.in_size
        self.out_size = LinOp.out_size

    def apply(self, x):
        return self.LinOp.apply(x) + self.scalar

    def applyAdjoint(self, x):
        return self.LinOp.applyAdjoint(x)

class LinOpDiff(BaseLinOp):
    def __init__(self, LinOp1, LinOp2):
        self.LinOp1 = LinOp1
        self.LinOp2 = LinOp2
        self.in_size = np.maximum(LinOp1.in_size, LinOp2.in_size)
        self.out_size = np.maximum(LinOp1.out_size, LinOp2.out_size)

    def apply(self, x):
        return self.LinOp1.apply(x) - self.LinOp2.apply(x)

    def applyAdjoint(self, x):
        return self.LinOp1.applyAdjoint(x) - self.LinOp2.applyAdjoint(x)

class LinOpScalarDiff(BaseLinOp):
    def __init__(self, LinOp, other):
        self.LinOp = LinOp
        self.scalar = other
        self.in_size = LinOp.in_size
        self.out_size = LinOp.out_size

    def apply(self, x):
        return self.LinOp.apply(x) - self.scalar

    def applyAdjoint(self, x):
        return self.LinOp.applyAdjoint(x)
    
class LinOpScalarMul(BaseLinOp):
    def __init__(self, LinOp, other):
        self.LinOp = LinOp
        self.scalar = other
        self.in_size = LinOp.in_size
        self.out_size = LinOp.out_size

    def apply(self, x):
        return self.LinOp.apply(x) * self.scalar

    def applyAdjoint(self, x):
        return self.LinOp.applyAdjoint(x) * self.scalar
    
class StackLinOp(BaseLinOp):
    def __init__(self, LinOp1, LinOp2):
        self.LinOp1 = LinOp1
        self.LinOp2 = LinOp2
        self.in_size = np.maximum(LinOp1.in_size, LinOp2.in_size)
        self.out_size = LinOp1.out_size + LinOp2.out_size  # TODO: handle cases where out_size is undefined

    def apply(self, x):
        return torch.cat((self.LinOp1.apply(x), self.LinOp2.apply(x)), dim=0)

    def applyAdjoint(self, x):
        return self.LinOp2.applyAdjoint(x[:self.LinOp2.out_size]) + self.LinOp1.applyAdjoint(x[self.LinOp2.out_size:])

