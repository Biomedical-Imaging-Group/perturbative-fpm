import torch as th
import pyphaseretrieve.linop as pl


n_shifts = 10
n, c, h, w = 2, 1, 50, 50
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


for h in range(20, 30):
    for w in range(20, 30):
        for hdiff in range(10):
            for wdiff in range(10):
                crop = pl.Crop2((h, w), (h - hdiff, w - wdiff))
                x = th.randn((n, c, h, w), dtype=th.float64)
                y = th.randn((n, c, h - hdiff, w - wdiff), dtype=th.float64)
                assert th.allclose(
                    ((crop @ x) * y).sum(),
                    (x * (crop.T @ y)).sum(),
                )
