import torch as th
import pyphaseretrieve.phaseretrieval as pp
import pyphaseretrieve.linop as plinop


def gradient_descent(
    grad,
    tau: float,
    x0: th.Tensor,
    n_iter: int = 100,
    callback=lambda x: None,
):
    x = x0.clone()
    for _ in range(n_iter):
        x -= tau * grad(x)
        callback(x)
    return x


def gauss_newton(
    f,
    x0,
    n_iter,
    solve,
    callback=lambda x: None,
):
    x = x0.clone()
    dx = th.zeros_like(x)

    for _ in range(n_iter):
        J = f.jacobian(x)
        dx = solve(J.T @ J, J.T @ f(x))
        x += dx
        callback(x)

    return x


# TODO implement batched
def conjugate_gradient(
    A: plinop.LinOp,
    b: th.Tensor,
    x0: th.Tensor,
    n_iter: int = 100,
    tol: float = 1e-9,
    dim: tuple[int, ...] = (1, 2, 3),
    callback=lambda x: None,
):
    def inner(a, b):
        return (a.conj() * b).sum(dim=dim).real

    x = x0.clone()
    r = b - A @ x
    p = r
    for _ in range(n_iter):
        Ap = A @ p
        alpha = inner(r, r) / inner(p, Ap)
        x = x + alpha * p
        r_new = r - alpha * Ap
        beta = inner(r_new, r_new) / inner(r, r)
        p = r_new + beta * p
        r = r_new.clone()
        callback(x)

        if (th.sqrt((r.abs()**2).sum(dim=dim)) < tol).all():
            break
    return x


def irgn(
    f,
    x0,
    n_iter,
    solve,
    callback=lambda x: None,
):
    x = x0.clone()
    dx = th.zeros_like(x)

    for _ in range(n_iter):
        J = f.jacobian(x)
        dx = solve(J, x)
        x += dx
        callback(x)

    return x


def condat_vu(K, prox_g, prox_fs, nabla_h, tau, sigma, x_0, y_0, callback=lambda x, y, i: None, n_iter=100):
    x = x_0.clone()
    y = y_0.clone()

    for _ in range(n_iter):
        x_old = x.clone()
        x = prox_g(x - tau * (K.T @ y + nabla_h(x)))
        y = prox_fs(y + sigma * (K @ (2 * x - x_old)))
        callback(x, y, _)

    return x



def power_iteration(A, x0, n_iter=10):
    x = x0.clone()

    for _ in range(n_iter):
        ax = A @ x
        x = ax / (ax.abs() ** 2).sum().sqrt()

    return x



def gerchberg_saxton(
    near_field_intensity: th.Tensor,
    far_field_intensity: th.Tensor,
    x0: th.Tensor,
    n_iter: int = 100,
):
    x = x0.clone()

    amp_fourier_space = th.sqrt(far_field_intensity)
    amp_real_space = th.sqrt(near_field_intensity)
    field_real_space = amp_real_space * th.exp(1j * x0)
    for i_iter in range(n_iter):
        field_fourier_space = amp_fourier_space * th.exp(
            1j * th.angle(th.fft.fft2(field_real_space))
        )

        field_real_space = amp_real_space * th.exp(
            1j * th.angle(th.fft.ifft2(field_fourier_space))
        )

    return field_real_space
