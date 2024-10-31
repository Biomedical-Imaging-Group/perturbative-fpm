import torch as th
import pyphaseretrieve.phaseretrieval as pp
import pyphaseretrieve.linop as plinop


class SpectralMethod:
    def __init__(self, pr_model):
        self.pr_model = pr_model
        self.x_shape = pr_model.in_shape

    def iterate(self, y, initial_est=None, n_iter=100, method="Lu"):
        y_norm = y / th.mean(y)
        if method == "Lu":
            threshold = th.maximum(1 - 1 / y_norm, -1)
        else:
            threshold = y_norm

        if initial_est is not None:
            x_est = th.copy(initial_est)
        else:
            x_est = th.random.randn(*self.x_shape)

        for i_iter in range(th.minimum(n_iter, 10)):
            x_new = self.pr_model.apply(x_est)
            x_new = threshold * x_new
            x_new = self.pr_model.applyT(x_new)
            x_est = x_new / th.linalg.norm(x_new)

        x_new = self.pr_model.apply(x_est)
        x_new = threshold * x_new
        x_new = self.pr_model.applyT(x_new)
        corr = th.real(x_new.ravel().T.conj() @ x_est.ravel())

        if corr < 0:
            for i_iter in range(n_iter):
                x_new = self.pr_model.apply(x_est)
                x_new = threshold * x_new
                x_new = self.pr_model.applyT(x_new)
                x_new = x_new + 1.1 * th.abs(corr) * x_est
                x_est = x_new / th.linalg.norm(x_new)
        else:
            for i_iter in range(n_iter - 10):
                x_new = self.pr_model.apply(x_est)
                x_new = threshold * x_new
                x_new = self.pr_model.applyT(x_new)
                x_est = x_new / th.linalg.norm(x_new)
        return x_est


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
