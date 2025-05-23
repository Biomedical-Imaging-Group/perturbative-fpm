import numpy as np
import torch as th
from pyphaseretrieve.base_linop import LinOp


class Matrix(LinOp):
    def __init__(self, matrix):
        self.H = matrix
        self.in_shape = (matrix.shape[1],)
        self.out_shape = (matrix.shape[0],)

    def apply(self, x):
        return self.H @ x

    def applyT(self, y):
        return self.H.T.conj() @ y


class Mul(LinOp):
    """coefs is for element-wise multiplication"""

    def __init__(self, coefs):
        self.coefs = coefs
        self.in_shape = coefs.shape
        self.out_shape = coefs.shape

    def apply(self, x):
        return self.coefs * x

    def applyT(self, y):
        return self.coefs.conj() * y


class Fft(LinOp):
    def __init__(self):
        self.in_shape = (-1,)
        self.out_shape = (-1,)

    def apply(self, x):
        return th.fft.fft(x, norm="ortho")

    def applyT(self, y):
        return th.fft.ifft(y, norm="ortho")


class Ifft(LinOp):
    def __init__(self):
        self.in_shape = (-1,)
        self.out_shape = (-1,)

    def apply(self, x):
        return th.fft.ifft(x, norm="ortho")

    def applyT(self, y):
        return th.fft.fft(y, norm="ortho")


class Fftshift(LinOp):
    def __init__(self):
        self.in_shape = (-1,)
        self.out_shape = (-1,)

    def apply(self, x):
        return th.fft.fftshift(x, dim=(-2, -1))

    def applyT(self, y):
        return th.fft.ifftshift(y, dim=(-2, -1))


class Ifftshift(LinOp):
    def __init__(self):
        self.in_shape = (-1,)
        self.out_shape = (-1,)

    def apply(self, x):
        return th.fft.ifftshift(x, dim=(-2, -1))

    def applyT(self, y):
        return th.fft.fftshift(y, dim=(-2, -1))


class Id(LinOp):
    def __init__(self):
        self.in_shape = (-1,)
        self.out_shape = (-1,)

    def apply(self, x):
        return x

    def applyT(self, y):
        return y


class Flip(LinOp):
    def __init__(self):
        self.in_shape = (-1,)
        self.out_shape = (-1,)

    def apply(self, x):
        return th.flip(x)

    def applyT(self, y):
        return th.flip(y)


class Roll(LinOp):
    def __init__(self, shifts):
        self.in_shape = (-1,)
        self.out_shape = (-1,)
        self.shifts = shifts.to(th.int64)

    def apply(self, x):
        n, _, h, w = x.shape
        c = self.shifts.shape[0]
        expanded = x.expand(-1, c, -1, -1)
        # https://discuss.pytorch.org/t/tensor-shifts-in-torch-roll/170655/2
        # This is still not really optimal, lots of stuff done for nothing
        ind0 = th.arange(n, dtype=th.int64)[:, None, None, None].expand(n, c, h, w)
        ind1 = th.arange(c, dtype=th.int64)[None, :, None, None].expand(n, c, h, w)
        ind2 = th.arange(h, dtype=th.int64)[None, None, :, None].expand(n, c, h, w)
        ind3 = th.arange(w, dtype=th.int64)[None, None, None, :].expand(n, c, h, w)

        return expanded[
            ind0,
            ind1,
            (ind2 + self.shifts[:, 0, None, None, None]) % h,
            (ind3 + self.shifts[:, 1, None, None, None]) % w,
        ]

    def applyT(self, y):
        n, _, h, w = y.shape
        c = self.shifts.shape[0]
        expanded = y.expand(-1, c, -1, -1)
        # https://discuss.pytorch.org/t/tensor-shifts-in-torch-roll/170655/2
        # This is still not really optimal, lots of stuff done for nothing
        ind0 = th.arange(n)[:, None, None, None].expand(n, c, h, w)
        ind1 = th.arange(c)[None, :, None, None].expand(n, c, h, w)
        ind2 = th.arange(h)[None, None, :, None].expand(n, c, h, w)
        ind3 = th.arange(w)[None, None, None, :].expand(n, c, h, w)

        return expanded[
            ind0,
            ind1,
            (ind2 - self.shifts[:, 0, None, None, None]) % h,
            (ind3 - self.shifts[:, 1, None, None, None]) % w,
        ]


class Real(LinOp):
    def __init__(self):
        self.in_shape = (-1,)
        self.out_shape = (-1,)

    def apply(self, x):
        return x.real

    def applyT(self, y):
        return y


class Imag(LinOp):
    def __init__(self):
        self.in_shape = (-1,)
        self.out_shape = (-1,)

    def apply(self, x):
        return x.imag

    def applyT(self, y):
        return y


# https://chatgpt.com/share/671f9a7c-8ccc-8011-a1f1-57c74c2e1db1
class RealPartExpand(LinOp):
    def __init__(self):
        # TODO this is incorrect
        self.in_shape = (-1,)
        self.out_shape = (-1,)

    def apply(self, x):
        return x.real

    def applyT(self, y):
        return y + 0j


# 2D classes
class Matrix2(LinOp):
    def __init__(self, matrix):
        self.H = matrix
        self.in_shape = (-1,)
        self.out_shape = (-1,)

    def apply(self, x):
        return self.H @ x

    def applyT(self, y):
        return self.H.T.conj() @ y


class Fft2(LinOp):
    def __init__(self):
        self.in_shape = (-1,)
        self.out_shape = (-1,)

    def apply(self, x):
        return th.fft.fft2(x, norm="ortho")

    def applyT(self, y):
        return th.fft.ifft2(y, norm="ortho")


class Ifft2(LinOp):
    def __init__(self):
        self.in_shape = (-1,)
        self.out_shape = (-1,)

    def apply(self, x):
        return th.fft.ifft2(x, norm="ortho")

    def applyT(self, y):
        return th.fft.fft2(y, norm="ortho")


class Roll2(LinOp):
    def __init__(self, shifts):
        self.in_shape = (-1,)
        self.out_shape = (-1,)
        self.shifts = shifts.to(th.int64)

    def apply(self, x):
        n, _, h, w = x.shape
        c = self.shifts.shape[0]
        expanded = x.expand(-1, c, -1, -1)
        # TODO
        # https://discuss.pytorch.org/t/tensor-shifts-in-torch-roll/170655/2
        # This is still not really optimal, lots of stuff done for nothing
        # I think we can make this more efficient with a double-gather
        self.ind0 = th.arange(n, dtype=th.int64, device=x.device)[
            :, None, None, None
        ].expand(n, c, h, w)
        self.ind1 = th.arange(c, dtype=th.int64, device=x.device)[
            None, :, None, None
        ].expand(n, c, h, w)
        self.ind2 = th.arange(h, dtype=th.int64, device=x.device)[
            None, None, :, None
        ].expand(n, c, h, w)
        self.ind3 = th.arange(w, dtype=th.int64, device=x.device)[
            None, None, None, :
        ].expand(n, c, h, w)

        return expanded[
            self.ind0,
            self.ind1,
            (self.ind2 + self.shifts[None, :, 0, None, None]) % h,
            (self.ind3 + self.shifts[None, :, 1, None, None]) % w,
        ]

    def applyT(self, y):
        return y[
            self.ind0,
            self.ind1,
            (self.ind2 - self.shifts[None, :, 0, None, None]) % y.shape[2],
            (self.ind3 - self.shifts[None, :, 1, None, None]) % y.shape[3],
        ].sum(1, keepdim=True)


class PhaseShift(LinOp):
    def __init__(self, shifts, shape):
        self.in_shape = (-1,)
        self.out_shape = (-1,)
        M, N = shape
        x = th.arange(M)[:, None].to(shifts.device) / M
        y = th.arange(N)[None, :].to(shifts.device) / N
        self.phase_shift = th.exp(
            -2j
            * np.pi
            * (x[None] * shifts[:, 0, None, None] + y[None] * shifts[:, 1, None, None])
        )[None]

    def apply(self, x):
        return x * self.phase_shift

    def applyT(self, y):
        return (y * self.phase_shift.conj()).sum(1, keepdim=True)


class ShiftInterp(LinOp):
    def __init__(self, shifts):
        self.in_shape = (-1,)
        self.out_shape = (-1,)
        self.shifts = shifts.fliplr()
        self.dtype = th.float32

    def apply(self, x):
        x_ = x.expand(self.shifts.shape[0], 1, *x.shape[2:])
        thetas = th.eye(2, 3, dtype=th.float32)[None].repeat(self.shifts.shape[0], 1, 1)
        thetas[:, :, 2] = self.shifts / x.shape[2] * 2.0
        grid = (
            th.nn.functional.affine_grid(
                thetas, (self.shifts.shape[0], 1, *x.shape[2:]), align_corners=False
            )
            .to(self.dtype)
            .to(x.device)
        )
        grid = ((grid + 1) % 2) - 1
        return (
            th.nn.functional.grid_sample(x_.real, grid, align_corners=False)
            + 1j * th.nn.functional.grid_sample(x_.imag, grid, align_corners=False)
        ).permute(1, 0, 2, 3)

    def applyT(self, y):
        y_ = y.permute(1, 0, 2, 3)
        thetas = th.eye(2, 3, dtype=th.float32)[None].repeat(self.shifts.shape[0], 1, 1)
        thetas[:, :, 2] = -self.shifts / y.shape[2] * 2
        grid = (
            th.nn.functional.affine_grid(
                thetas, (self.shifts.shape[0], 1, *y.shape[2:]), align_corners=False
            )
            .to(self.dtype)
            .to(y.device)
        )
        grid = ((grid + 1) % 2) - 1
        return (
            th.nn.functional.grid_sample(y_.real, grid, align_corners=False)
            + 1j * th.nn.functional.grid_sample(y_.imag, grid, align_corners=False)
        ).sum(0, keepdim=True)


# TODO tests
class Crop2(LinOp):
    def __init__(self, in_shape, crop_shape):
        self.in_shape = in_shape
        self.out_shape = crop_shape

        self.istart = (self.in_shape[0] - self.out_shape[0]) // 2
        self.iend = self.istart + self.out_shape[0]

        self.jstart = (self.in_shape[1] - self.out_shape[1]) // 2
        self.jend = self.jstart + self.out_shape[1]

        ipad2 = self.in_shape[0] - self.iend
        jpad2 = self.in_shape[1] - self.jend

        self.pads = tuple(int(pad) for pad in (self.jstart, jpad2, self.istart, ipad2))

    def apply(self, x):
        return x[:, :, self.istart : self.iend, self.jstart : self.jend]

    def applyT(self, y):
        return th.nn.functional.pad(y, self.pads, mode="constant")


class Roll2_PadZero(LinOp):
    def __init__(self, v_shifts, h_shifts):
        self.in_shape = (-1,)
        self.out_shape = (-1,)
        self.v_shifts = int(v_shifts)
        self.h_shifts = int(h_shifts)

    def apply(self, x):
        x = th.roll(x, self.h_shifts, dims=1)
        if self.h_shifts < 0:
            x[:, self.h_shifts :] = 0
        elif self.h_shifts > 0:
            x[:, 0 : self.h_shifts] = 0

        x = th.roll(x, self.v_shifts, dims=0)
        if self.v_shifts < 0:
            x[self.v_shifts :, :] = 0
        elif self.v_shifts > 0:
            x[0 : self.v_shifts, :] = 0
        return x

    def applyT(self, y):
        y = th.roll(y, -self.h_shifts, dims=1)
        if -self.h_shifts < 0:
            y[:, -self.h_shifts :] = 0
        elif -self.h_shifts > 0:
            y[:, 0 : -self.h_shifts] = 0

        y = th.roll(y, -self.v_shifts, dims=0)
        if -self.v_shifts < 0:
            y[-self.v_shifts :, :] = 0
        elif -self.v_shifts > 0:
            y[0 : -self.v_shifts, :] = 0
        return y


# Dimensionless
# TODO this is a misnomer; at this point it assumes that the input has 'channel'
# dimension of size 1 and puts all outputs into the channel dimension
class Stack(LinOp):
    def __init__(self, linops):
        self.linops = linops
        # TODO
        self.in_shape = (0, 0)
        self.out_shape = (0, 0)

    def apply(self, x):
        return th.cat(tuple(linop.apply(x) for linop in self.linops), dim=1)

    # TODO this is not generic yet... probably the best implementation is just
    # with using lists..
    def applyT(self, y):
        res = self.linops[0].applyT(y[:, 0:1, :, :])
        for idx, linop in enumerate(self.linops[1:], start=1):
            res += linop.applyT(y[:, idx : idx + 1, :, :])
        return res


# functions
def shift_2d_replace(data, dx, dy, constant=False):
    shifted_data = th.roll(data, dx, dims=1)
    if dx < 0:
        shifted_data[:, dx:] = constant
    elif dx > 0:
        shifted_data[:, 0:dx] = constant

    shifted_data = th.roll(shifted_data, dy, dims=0)
    if dy < 0:
        shifted_data[dy:, :] = constant
    elif dy > 0:
        shifted_data[0:dy, :] = constant
    return shifted_data


class SumReduce(LinOp):
    def __init__(self, dim, size):
        self.in_shape = (-1,)
        self.out_shape = (-1,)
        self.dim = dim
        self.size = size

    def apply(self, x):
        return x.sum(dim=self.dim, keepdim=True)

    # TODO this is not generic yet wrt dimension
    def applyT(self, y):
        return y.expand(y.shape[0], self.size, *y.shape[2:])


class Grad(LinOp):
    def __init__(self):
        self.in_shape = (-1,)
        self.out_shape = (-1,)

    def apply(self, x):
        grad = th.zeros((x.shape[0], 2, *x.shape[2:]), device=x.device, dtype=x.dtype)
        grad[:, 0:1, :, :-1] += x[:, :, :, 1:] - x[:, :, :, :-1]
        grad[:, 1:2, :-1, :] += x[:, :, 1:, :] - x[:, :, :-1, :]
        return grad

    def applyT(self, y):
        div = th.zeros((y.shape[0], 1, *y.shape[2:]), device=y.device, dtype=y.dtype)
        div[:, :, :, 1:] += y[:, 0, :, :-1]
        div[:, :, :, :-1] -= y[:, 0, :, :-1]
        div[:, :, 1:, :] += y[:, 1, :-1, :]
        div[:, :, :-1, :] -= y[:, 1, :-1, :]
        return div
