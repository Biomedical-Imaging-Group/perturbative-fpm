from abc import abstractmethod

## Base class

class BaseLinOp:
    def __init__(self):
        self.in_shape:tuple
        self.out_shape:tuple

    @abstractmethod
    def apply(self, x):
        pass

    @abstractmethod
    def applyT(self, x):
        pass

    def __add__(self, other):
        if isinstance(other, BaseLinOp):
            return LinOpSum(self, other)
        else:
            raise NameError('Summing scalar and LinOp objects does not result in a linear operator.')

    def __radd__(self, other):
        if isinstance(other, BaseLinOp):
            return LinOpSum(self, other)
        else:
            raise NameError('Summing scalar and LinOp objects does not result in a linear operator.')
    
    def __sub__(self, other):
        if isinstance(other, BaseLinOp):
            return LinOpDiff(self, other)
        else:
            raise NameError('Subtracting scalar and LinOp objects does not result in a linear operator.')
    
    def __rsub__(self, other):
        if isinstance(other, BaseLinOp):
            return LinOpDiff(self, other)
        else:
            raise NameError('Subtracting scalar and LinOp objects does not result in a linear operator.')
        
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


## Utils classes
class LinOpComposition(BaseLinOp):
    def __init__(self, LinOp1:BaseLinOp, LinOp2:BaseLinOp):
        self.LinOp1 = LinOp1
        self.LinOp2 = LinOp2
        self.in_shape = LinOp1.in_shape if LinOp2.in_shape == (-1,) else LinOp2.in_shape       
        self.out_shape = LinOp2.out_shape if LinOp1.out_shape == (-1,) else LinOp1.out_shape   
        
    def apply(self, x):
        return self.LinOp1.apply(self.LinOp2.apply(x))

    def applyT(self, x):
        return self.LinOp2.applyT(self.LinOp1.applyT(x))

class LinOpSum(BaseLinOp):
    def __init__(self, LinOp1:BaseLinOp, LinOp2:BaseLinOp):
        self.LinOp1 = LinOp1
        self.LinOp2 = LinOp2
        self.in_shape = LinOp1.in_shape if LinOp1.in_shape > LinOp2.in_shape else LinOp2.in_shape
        self.out_shape = LinOp1.out_shape if LinOp1.out_shape > LinOp2.out_shape else LinOp2.out_shape

    def apply(self, x):
        return self.LinOp1.apply(x) + self.LinOp2.apply(x)

    def applyT(self, x):
        return self.LinOp2.applyT(x) + self.LinOp1.applyT(x)

class LinOpDiff(BaseLinOp):
    def __init__(self, LinOp1:BaseLinOp, LinOp2:BaseLinOp):
        self.LinOp1 = LinOp1
        self.LinOp2 = LinOp2
        self.in_shape = LinOp1.in_shape if LinOp1.in_shape > LinOp2.in_shape else LinOp2.in_shape
        self.out_shape = LinOp1.out_shape if LinOp1.out_shape > LinOp2.out_shape else LinOp2.out_shape

    def apply(self, x):
        return self.LinOp1.apply(x) - self.LinOp2.apply(x)

    def applyT(self, x):
        return self.LinOp1.applyT(x) - self.LinOp2.applyT(x)
    
class LinOpScalarMul(BaseLinOp):
    def __init__(self, LinOp:BaseLinOp, other):
        self.LinOp = LinOp
        self.scalar = other
        self.in_shape = LinOp.in_shape
        self.out_shape = LinOp.out_shape

    def apply(self, x):
        return self.LinOp.apply(x) * self.scalar

    def applyT(self, x):
        return self.LinOp.applyT(x) * self.scalar

class LinOpTranspose(BaseLinOp):
    def __init__(self, LinOpT:BaseLinOp):
        self.LinOpT = LinOpT
        self.in_shape = LinOpT.out_shape
        self.out_shape = LinOpT.in_shape

    def apply(self, x):
        return self.LinOpT.applyT(x)

    def applyT(self, x):
        return self.LinOpT.apply(x)