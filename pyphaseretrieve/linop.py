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
            raise NameError('Multiplying two LinOp objects does not result in a linear operator.')
        else:
            return LinOpScalarMul(self, other)
        
    def __rmul__(self, other):
        if isinstance(other, BaseLinOp):
            raise NameError('Multiplying two LinOp objects does not result in a linear operator.')
        else:
            return LinOpScalarMul(self, other)
    
    def __matmul__(self, other):
        if isinstance(other, BaseLinOp):
            return LinOpComposition(self, other)
        else:
            raise NameError('The matrix multiplication operator can only be performed between two LinOp objects.')

    def T(self):
        return LinOpTranspose(self)

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

class LinOpFFT(BaseLinOp):
    def __init__(self):
        self.in_size = -1
        self.out_size = -1

    def apply(self, x):
        return np.fft.fft(x, norm="ortho")

    def applyAdjoint(self, x):
        return np.fft.ifft(x, norm="ortho")

class LinOpFFT2(BaseLinOp):
    def __init__(self):
        self.in_size = -1
        self.out_size = -1

    def apply(self, x):
        return np.fft.fft2(x, norm="ortho")

    def applyAdjoint(self, x):
        return np.fft.ifft2(x, norm="ortho")

class LinOpId(BaseLinOp):
    def __init__(self):
        self.in_size = -1
        self.out_size = -1
        
    def apply(self, x):
        return x
    
    def applyAdjoint(self, x):
        return x
    
class LinOpConstant(BaseLinOp):
    """Warning: This is not a LinOp. Just here for convenience. """
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
        return np.flip(x)

    def applyAdjoint(self, x):
        return np.flip(x)

class LinOpRoll(BaseLinOp):
    def __init__(self, shifts, dims):
        self.in_size = -1
        self.out_size = -1
        self.shifts = shifts
        self.dims = dims

    def apply(self, x):
        return np.roll(x, shift=self.shifts, axis=self.dims)

    def applyAdjoint(self, x):
        return np.roll(x, shifts=(-shift for shift in self.shifts), axis=self.dims)

    
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
    def __init__(self, LinOpList):
        self.LinOpList = LinOpList
        self.in_size = np.amax(np.array([linop.in_size for linop in LinOpList]))
        self.out_size = np.sum(np.array([(linop.out_size if linop.out_size>0 else self.in_size)
            for linop in LinOpList]))

    def apply(self, x):
        return np.concatenate(tuple(linop.apply(x) for linop in self.LinOpList), axis=0)

    def applyAdjoint(self, x):
        current_idx = 0
        res = np.zeros(int(self.in_size))
        for linop in self.LinOpList:
            res += linop.applyAdjoint(x[current_idx:current_idx+linop.out_size])
            current_idx += (linop.out_size if linop.out_size>0 else self.in_size)
        return res

class LinOpTranspose(BaseLinOp):
    def __init__(self, LinOpT):
        self.LinOpT = LinOpT
        self.in_size = LinOpT.out_size
        self.out_size = LinOpT.in_size

    def apply(self, x):
        return self.LinOpT.applyAdjoint(x)

    def applyAdjoint(self, x):
        return self.LinOpT.apply(x)
