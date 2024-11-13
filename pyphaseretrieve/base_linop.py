from abc import abstractmethod
import torch as th


## Base class
class LinOp:
    def __init__(self):
        self.in_shape: tuple
        self.out_shape: tuple

    @abstractmethod
    def apply(self, x: th.Tensor) -> th.Tensor:
        pass

    @abstractmethod
    def applyT(self, y: th.Tensor) -> th.Tensor:
        pass

    def __add__(self, other):
        if isinstance(other, LinOp):
            return Sum(self, other)
        else:
            raise NameError(
                "Summing scalar and LinOp objects does not result in a linear operator."
            )

    def __radd__(self, other):
        if isinstance(other, LinOp):
            return Sum(self, other)
        else:
            raise NameError(
                "Summing scalar and LinOp objects does not result in a linear operator."
            )

    def __sub__(self, other):
        if isinstance(other, LinOp):
            return Diff(self, other)
        else:
            raise NameError(
                "Subtracting scalar and LinOp objects does not result in a linear operator."
            )

    def __rsub__(self, other):
        if isinstance(other, LinOp):
            return Diff(self, other)
        else:
            raise NameError(
                "Subtracting scalar and LinOp objects does not result in a linear operator."
            )

    def __mul__(self, other):
        if isinstance(other, LinOp):
            raise NameError(
                "Multiplying two LinOp objects does not result in a linear operator."
            )
        else:
            return ScalarMul(self, other)

    def __rmul__(self, other):
        if isinstance(other, LinOp):
            raise NameError(
                "Multiplying two LinOp objects does not result in a linear operator."
            )
        else:
            return ScalarMul(self, other)

    def __matmul__[T: (LinOp, th.Tensor)](self, other: T) -> T:
        if isinstance(other, LinOp):
            return Composition(self, other)
        return self.apply(other)

    @property
    def T(self):
        return Transpose(self)


## Utils classes
class Composition(LinOp):
    def __init__(self, LinOp1: LinOp, LinOp2: LinOp):
        self.LinOp1 = LinOp1
        self.LinOp2 = LinOp2
        self.in_shape = LinOp1.in_shape if LinOp2.in_shape == (-1,) else LinOp2.in_shape
        self.out_shape = (
            LinOp2.out_shape if LinOp1.out_shape == (-1,) else LinOp1.out_shape
        )

    def apply(self, x):
        return self.LinOp1.apply(self.LinOp2.apply(x))

    def applyT(self, y):
        return self.LinOp2.applyT(self.LinOp1.applyT(y))


class Sum(LinOp):
    def __init__(self, LinOp1: LinOp, LinOp2: LinOp):
        self.LinOp1 = LinOp1
        self.LinOp2 = LinOp2
        self.in_shape = (
            LinOp1.in_shape if LinOp1.in_shape > LinOp2.in_shape else LinOp2.in_shape
        )
        self.out_shape = (
            LinOp1.out_shape
            if LinOp1.out_shape > LinOp2.out_shape
            else LinOp2.out_shape
        )

    def apply(self, x):
        return self.LinOp1.apply(x) + self.LinOp2.apply(x)

    def applyT(self, y):
        return self.LinOp2.applyT(y) + self.LinOp1.applyT(y)


class Diff(LinOp):
    def __init__(self, LinOp1: LinOp, LinOp2: LinOp):
        self.LinOp1 = LinOp1
        self.LinOp2 = LinOp2
        self.in_shape = (
            LinOp1.in_shape if LinOp1.in_shape > LinOp2.in_shape else LinOp2.in_shape
        )
        self.out_shape = (
            LinOp1.out_shape
            if LinOp1.out_shape > LinOp2.out_shape
            else LinOp2.out_shape
        )

    def apply(self, x):
        return self.LinOp1.apply(x) - self.LinOp2.apply(x)

    def applyT(self, y):
        return self.LinOp1.applyT(y) - self.LinOp2.applyT(y)


class ScalarMul(LinOp):
    def __init__(self, LinOp: LinOp, other):
        self.LinOp = LinOp
        self.scalar = other
        self.in_shape = LinOp.in_shape
        self.out_shape = LinOp.out_shape

    def apply(self, x):
        return self.LinOp.apply(x) * self.scalar

    def applyT(self, y):
        return self.LinOp.applyT(y) * self.scalar


class Transpose(LinOp):
    def __init__(self, LinOpT: LinOp):
        self.LinOpT = LinOpT
        self.in_shape = LinOpT.out_shape
        self.out_shape = LinOpT.in_shape

    def apply(self, x):
        return self.LinOpT.applyT(x)

    def applyT(self, y):
        return self.LinOpT.apply(y)
