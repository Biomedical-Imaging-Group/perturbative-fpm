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

    def applyT(self, x):
        return self.H.T.conj() @ x


class Mul(LinOp):
    """coefs is for element-wise multiplication"""

    def __init__(self, coefs):
        self.coefs = coefs
        self.in_shape = coefs.shape
        self.out_shape = coefs.shape

    def apply(self, x):
        return self.coefs * x

    def applyT(self, x):
        return self.coefs.conj() * x


class Fft(LinOp):
    def __init__(self):
        self.in_shape = (-1,)
        self.out_shape = (-1,)

    def apply(self, x):
        return th.fft.fft(x, norm="ortho")

    def applyT(self, x):
        return th.fft.ifft(x, norm="ortho")


class Ifft(LinOp):
    def __init__(self):
        self.in_shape = (-1,)
        self.out_shape = (-1,)

    def apply(self, x):
        return th.fft.ifft(x, norm="ortho")

    def applyT(self, x):
        return th.fft.fft(x, norm="ortho")


class Fftshift(LinOp):
    def __init__(self):
        self.in_shape = (-1,)
        self.out_shape = (-1,)

    def apply(self, x):
        return th.fft.fftshift(x, dim=(-2, -1))

    def applyT(self, x):
        return th.fft.ifftshift(x, dim=(-2, -1))


class Ifftshift(LinOp):
    def __init__(self):
        self.in_shape = (-1,)
        self.out_shape = (-1,)

    def apply(self, x):
        return th.fft.ifftshift(x, dim=(-2, -1))

    def applyT(self, x):
        return th.fft.fftshift(x, dim=(-2, -1))


class Id(LinOp):
    def __init__(self):
        self.in_shape = (-1,)
        self.out_shape = (-1,)

    def apply(self, x):
        return x

    def applyT(self, x):
        return x


class Flip(LinOp):
    def __init__(self):
        self.in_shape = (-1,)
        self.out_shape = (-1,)

    def apply(self, x):
        return th.flip(x)

    def applyT(self, x):
        return th.flip(x)


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
        ind0 = th.arange(n, dtype=th.int64)[:, None, None, None].expand(
            n, c, h, w
        )
        ind1 = th.arange(c, dtype=th.int64)[None, :, None, None].expand(
            n, c, h, w
        )
        ind2 = th.arange(h, dtype=th.int64)[None, None, :, None].expand(
            n, c, h, w
        )
        ind3 = th.arange(w, dtype=th.int64)[None, None, None, :].expand(
            n, c, h, w
        )

        return expanded[
            ind0,
            ind1,
            (ind2 + self.shifts[:, 0, None, None, None]) % h,
            (ind3 + self.shifts[:, 1, None, None, None]) % w,
        ]

    def applyT(self, x):
        n, _, h, w = x.shape
        c = self.shifts.shape[0]
        expanded = x.expand(-1, c, -1, -1)
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

    def applyT(self, x):
        return x


class Imag(LinOp):
    def __init__(self):
        self.in_shape = (-1,)
        self.out_shape = (-1,)

    def apply(self, x):
        return x.imag

    def applyT(self, x):
        return x


# https://chatgpt.com/share/671f9a7c-8ccc-8011-a1f1-57c74c2e1db1
class RealPartExpand(LinOp):
    def __init__(self):
        # TODO this is incorrect
        self.in_shape = (-1, )
        self.out_shape = (-1, )

    def apply(self, x):
        return x.real

    def applyT(self, x):
        return x + 0j


# 2D classes
class Matrix2(LinOp):
    def __init__(self, matrix):
        self.H = matrix
        self.in_shape = (-1,)
        self.out_shape = (-1,)

    def apply(self, x):
        return self.H @ x

    def applyT(self, x):
        return self.H.T.conj() @ x


class Fft2(LinOp):
    def __init__(self):
        self.in_shape = (-1,)
        self.out_shape = (-1,)

    def apply(self, x):
        return th.fft.fft2(x, norm="ortho")

    def applyT(self, x):
        return th.fft.ifft2(x, norm="ortho")


class Ifft2(LinOp):
    def __init__(self):
        self.in_shape = (-1,)
        self.out_shape = (-1,)

    def apply(self, x):
        return th.fft.ifft2(x, norm="ortho")

    def applyT(self, x):
        return th.fft.fft2(x, norm="ortho")


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
        ind0 = th.arange(n, dtype=th.int64, device=x.device)[
            :, None, None, None
        ].expand(n, c, h, w)
        ind1 = th.arange(c, dtype=th.int64, device=x.device)[
            None, :, None, None
        ].expand(n, c, h, w)
        ind2 = th.arange(h, dtype=th.int64, device=x.device)[
            None, None, :, None
        ].expand(n, c, h, w)
        ind3 = th.arange(w, dtype=th.int64, device=x.device)[
            None, None, None, :
        ].expand(n, c, h, w)

        return expanded[
            ind0,
            ind1,
            (ind2 + self.shifts[None, :, 0, None, None]) % h,
            (ind3 + self.shifts[None, :, 1, None, None]) % w,
        ]

    def applyT(self, x):
        n, c, h, w = x.shape
        ind0 = th.arange(n, device=x.device)[:, None, None, None].expand(
            n, c, h, w
        )
        ind1 = th.arange(c, device=x.device)[None, :, None, None].expand(
            n, c, h, w
        )
        ind2 = th.arange(h, device=x.device)[None, None, :, None].expand(
            n, c, h, w
        )
        ind3 = th.arange(w, device=x.device)[None, None, None, :].expand(
            n, c, h, w
        )

        return x[
            ind0,
            ind1,
            (ind2 - self.shifts[None, :, 0, None, None]) % h,
            (ind3 - self.shifts[None, :, 1, None, None]) % w,
        ].sum(1, keepdim=True)


class PhaseShift(LinOp):
    def __init__(self, shifts, shape):
        M, N = shape
        x = th.arange(M).reshape(-1, 1).to(shifts.device)
        y = th.arange(N).reshape(1, -1).to(shifts.device)
        self.in_shape = (-1,)
        self.out_shape = (-1,)
        self.shifts = shifts
        self.phase_shift = th.exp(2j * np.pi * (x[None] * self.shifts[:, 0, None, None] / M + y[None] * self.shifts[:, 1, None, None] / N))[None]

    # TODO check correctness
    def apply(self, x):
        return x * self.phase_shift

    # TODO check correctness
    def applyT(self, y):
        return (y * self.phase_shift.conj()).sum(1, keepdim=True)


# TODO tests
class Crop2(LinOp):
    def __init__(self, in_shape, crop_shape):
        """assume square size of input image"""
        self.in_shape = in_shape
        self.out_shape = crop_shape

        self.vstart = int(self.in_shape[0] // 2 - self.out_shape[0] // 2)
        self.vend = self.vstart + self.out_shape[0]
        self.hstart = int(self.in_shape[1] // 2 - self.out_shape[1] // 2)
        self.hend = self.hstart + self.out_shape[1]

        v_pad_size = self.in_shape[0] - self.out_shape[0]
        h_pad_size = self.in_shape[1] - self.out_shape[1]

        if self.in_shape[0] % 2 == 1:
            vpad1 = np.floor(v_pad_size / 2)
            vpad2 = np.ceil(v_pad_size / 2)
        else:
            vpad1 = np.ceil(v_pad_size / 2)
            vpad2 = np.floor(v_pad_size / 2)

        if self.in_shape[1] % 2 == 1:
            hpad1 = np.floor(h_pad_size / 2)
            hpad2 = np.ceil(h_pad_size / 2)
        else:
            hpad1 = np.ceil(h_pad_size / 2)
            hpad2 = np.floor(h_pad_size / 2)

        self.pads = tuple(int(pad) for pad in (vpad1, vpad2, hpad1, hpad2))

    def apply(self, x):
        return x[:, :, self.vstart:self.vend, self.hstart:self.hend]

    def applyT(self, x):
        return th.nn.functional.pad(x, self.pads, mode="constant")


class Roll2_PadZero(LinOp):
    def __init__(self, v_shifts, h_shifts):
        self.in_shape = (-1,)
        self.out_shape = (-1,)
        self.v_shifts = int(v_shifts)
        self.h_shifts = int(h_shifts)

    def apply(self, x):
        x = th.roll(x, self.h_shifts, dims=1)
        if self.h_shifts < 0:
            x[:, self.h_shifts:] = 0
        elif self.h_shifts > 0:
            x[:, 0: self.h_shifts] = 0

        x = th.roll(x, self.v_shifts, dims=0)
        if self.v_shifts < 0:
            x[self.v_shifts:, :] = 0
        elif self.v_shifts > 0:
            x[0: self.v_shifts, :] = 0
        return x

    def applyT(self, x):
        x = th.roll(x, -self.h_shifts, dims=1)
        if -self.h_shifts < 0:
            x[:, -self.h_shifts:] = 0
        elif -self.h_shifts > 0:
            x[:, 0: -self.h_shifts] = 0

        x = th.roll(x, -self.v_shifts, dims=0)
        if -self.v_shifts < 0:
            x[-self.v_shifts:, :] = 0
        elif -self.v_shifts > 0:
            x[0: -self.v_shifts, :] = 0
        return x


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
    def applyT(self, x):
        res = self.linops[0].applyT(x[:, 0:1, :, :])
        for idx, linop in enumerate(self.linops[1:], start=1):
            res += linop.applyT(x[:, idx: idx + 1, :, :])
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
    def applyT(self, x):
        return x.expand(x.shape[0], self.size, *x.shape[2:])
