import torch as th
import pyphaseretrieve.linop as pl


n_shifts = 10
n, c, h, w = 2, 1, 512, 512
low = -20
high = 20
shifts = th.randint(low, high, (n_shifts, 2))
roll = pl.Roll2(shifts)

x = th.randn((n, c, h, w))
y = th.randn((n, n_shifts, h, w))

assert th.allclose(
    ((roll @ x) * y).sum(),
    (x * (roll.T @ y)).sum(),
)

c = 10
sumreduce = pl.SumReduce(dim=1, size=c)
x = th.randn((n, c, h, w))
y = th.randn((n, 1, h, w))

assert th.allclose(
    ((sumreduce @ x) * y).sum(),
    (x * (sumreduce.T @ y)).sum(),
)

real = pl.RealPartExpand()
x = th.randn((n, c, h, w), dtype=th.complex64)
y = th.randn((n, c, h, w), dtype=th.float32)

assert th.allclose(
    ((real @ x) * y).sum(),
    (x * (real.T @ y)).real.sum(),
)

x = th.randn((n, 1, h, w), dtype=th.complex64)
y = th.randn((n, c, h, w), dtype=th.complex64)
shifts = th.randint(-10, 10, (c, 2))
phaseshift = pl.PhaseShift(shifts, (h, w))
assert th.allclose(
    ((phaseshift @ x) * y.conj()).sum(),
    (x * (phaseshift.T @ y).conj()).sum(),
)


# for h in range(20, 30):
#     for w in range(20, 30):
#         for hdiff in range(10):
#             for wdiff in range(10):
#                 crop = pl.Crop2((h, w), (h - hdiff, w - wdiff))
#                 x = th.randn((n, c, h, w), dtype=th.float64)
#                 y = th.randn((n, c, h - hdiff, w - wdiff), dtype=th.float64)
#                 assert th.allclose(
#                     ((crop @ x) * y).sum(),
#                     (x * (crop.T @ y)).sum(),
#                 )


# Here we need n=1, this doesnt work batches unfortunately
# n = 1
# x = th.randn((n, 1, h, w), dtype=th.complex128)
# y = th.randn((n, c, h, w), dtype=th.complex128)
# shifts = th.randint(-10, 10, (c, 2))
# phaseshift = pl.ShiftInterp(shifts)
# assert th.allclose(
#     ((phaseshift @ x) * y.conj()).sum(),
#     (x * (phaseshift.T @ y).conj()).sum(),
# )

# h, w = 220, 220
# x = th.randn((1, 1, h, w))
# shifts = th.randint(low*10, high*10, (n_shifts, 2))

# import skimage
# x = th.from_numpy(skimage.data.camera()[1:, 1:][None, None]) / 255. - .5
# x = th.from_numpy(skimage.data.camera()[None, None]) / 255. - .5

test_roll = False
if test_roll:
    x_ = x.expand(n_shifts, 1, *x.shape[2:])
    theta = th.eye(2, 3, dtype=th.float64)[None].repeat(shifts.shape[0], 1, 1)
    theta[:, :, 2] = shifts.fliplr() * 2.0 / x.shape[2]
    grid = (
        th.nn.functional.affine_grid(
            theta, (n_shifts, 1, *x.shape[2:]), align_corners=False
        )
        .to(th.float32)
        .to(x.device)
    )
    sampled = th.nn.functional.grid_sample(x_, grid, align_corners=False).permute(
        1, 0, 2, 3
    )

    roll = pl.Roll2(th.round(shifts).to(th.int32))
    rolled = roll @ x.real[0:1]

    import matplotlib.pyplot as plt
    import numpy as np

    for sh, s, r in zip(shifts, sampled[0], rolled[0]):
        print(sh)
        plt.figure()
        plt.imshow(s.numpy())
        plt.figure()
        plt.imshow(r.numpy())
        plt.figure()
        plt.imshow(np.abs(s.numpy() - r.numpy()))
        plt.show()

import matplotlib.pyplot as plt

test_prox = True
n_shifts = 3
shifts = th.randint(-10, 10, (n_shifts, 2))
ps = 30
probe = th.randint(0, 2, (ps, ps))
b = th.randn((1, n_shifts, ps, ps), dtype=th.complex128)
x_bar = th.randn((1, 1, h, w), dtype=th.complex128)
pad = (h - ps) // 2
probe_padded_shifted = probe
roll = pl.Roll2(th.round(shifts).to(th.int64))
crop = pl.Ifftshift() @ pl.Crop2(x_bar.shape[2:], probe.shape) @ pl.Fftshift()

# Prox of 'forward'
if test_prox:
    A = pl.Ifft2() @ pl.Mul(probe) @ crop @ roll @ pl.Fft2()
    div = 1 + roll.T @ crop.T @ pl.Mul(
        probe_padded_shifted
    ) @ crop @ roll @ th.ones_like(x_bar)
    xs = pl.Ifft2() @ pl.Mul(1 / div) @ pl.Fft2() @ (x_bar + A.T @ b)
    lhs = (A.T @ A + pl.Id()) @ xs
    rhs = x_bar + A.T @ b
    plt.figure()
    plt.imshow((lhs.imag / rhs.imag).abs()[0, 0].clip(0, 100))

# TODO Prox of jacobian (possible?)
field = A @ x_bar
if test_prox:
    A = (
        pl.SumReduce(1, n_shifts)
        @ pl.RealPartExpand()
        @ pl.Mul(2 * field.conj())
        @ pl.Ifft2()
        @ pl.Mul(probe)
        @ crop
        @ roll
        @ pl.Fft2()
    )
    div = 1 + roll.T @ crop.T @ pl.Mul(
        probe_padded_shifted
    ) @ crop @ roll @ th.ones_like(x_bar)
    xs = pl.Ifft2() @ pl.Mul(1 / div) @ pl.Fft2() @ (x_bar + A.T @ b)
    lhs = (A.T @ A + pl.Id()) @ xs
    rhs = x_bar + A.T @ b
    plt.figure()
    plt.imshow((lhs.imag / rhs.imag).abs()[0, 0].clip(0, 100))

plt.show()
