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

# TODO check if this is correct more thoroughly
# TODO revert to angles again instead of shifts!
x = th.randn((n, 1, h, w), dtype=th.complex64)
y = th.randn((n, c, h, w), dtype=th.complex64)
shifts = th.randint(-10, 10, (c, 2))
phaseshift = pl.PhaseShift(shifts)
assert th.allclose(
    ((phaseshift @ x) * y.conj()).sum(),
    (x * (phaseshift.T @ y).conj()).sum(),
)
