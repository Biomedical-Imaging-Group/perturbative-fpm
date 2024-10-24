import torch as th
import pyphaseretrieve.loss as pl
import pyphaseretrieve.phaseretrieval as pp
import pyphaseretrieve.linop as plinop


class GradientDescent:
    def __init__(
        self,
        pr_model,
        loss_func: pl.LossFunction | None = None,
        line_search=None,
        acceleration=None,
    ):
        self.pr_model = pr_model

        if loss_func is None:
            self.loss_func = pl.loss_intensity_based()
        else:
            self.loss_func = loss_func

        self.line_search = line_search
        self.acceleration = acceleration

        self.x_shape = pr_model.in_shape
        self.loss_list = []

    def iterate(self, y, initial_est=None, n_iter=100, lr=1, alpha=0):
        x_est = initial_est.clone()

        for i_iter in range(n_iter):
            loss, grad = self.loss_func.compute_loss(y, self.pr_model, x_est)
            self.loss_list.append(loss)

            if self.acceleration is not None:
                descent_direction = self.find_dir(x_est, y, grad)
            else:
                descent_direction = -grad

            if self.line_search is not None:
                actual_lr = self.find_lr(
                    x_est, y, descent_direction, grad, loss, initial_lr=lr
                )
            else:
                actual_lr = lr
            x_est += actual_lr * descent_direction
        return x_est

    def iterate_local_GradientDescent(
        self, y, initial_est=None, n_iter=100, lr=1
    ):
        if initial_est is not None:
            x_est = th.copy(initial_est)
        else:
            x_est = th.ones(shape=self.x_shape, dtype=th.complex128)

        pr_model_linop_list = self.pr_model.get_linop_list()

        for i_iter in range(n_iter):
            loss = self.loss_func.compute_loss(
                y, self.pr_model, x_est, compute_grad=False
            )
            self.loss_list.append(loss)

            current_idx = 0
            for idx, i_linop in enumerate(pr_model_linop_list):
                _, grad = self.loss_func.compute_loss(
                    y[
                        current_idx : current_idx
                        + self.pr_model.probe_shape[0],
                        :,
                    ],
                    pp.FourierPtychography(i_linop),
                    x_est,
                )
                x_est += lr * (-grad)
                current_idx += self.pr_model.probe_shape[0]
        return x_est

    def find_lr(
        self,
        x_est,
        y,
        descent_direction,
        current_grad,
        initial_loss,
        initial_lr=1,
        c=0.9,
        tau=0.5,
    ):
        lr = initial_lr
        m = th.real(current_grad.ravel().T.conj() @ descent_direction.ravel())
        if m >= 0:
            print(
                "There may be a sign error in the computation of the descent direction."
            )
        while True:
            new_x_est = x_est + lr * descent_direction
            new_y_est = th.abs(self.pr_model.apply(new_x_est)) ** 2
            new_loss = th.sum((new_y_est - y) ** 2)
            if new_loss <= initial_loss + lr * c * m:
                break
            lr = tau * lr
        return lr

    def find_dir(self, x_est, y, grad):
        if self.acceleration == "conjugate gradient":
            if not hasattr(self, "previous_grad"):
                self.previous_grad = grad
                self.previous_direction = -grad
                return -grad
            else:
                beta = th.maximum(
                    0,
                    th.real(
                        grad.ravel().conj().T
                        @ (grad.ravel() - self.previous_grad.ravel())
                        / (
                            self.previous_grad.ravel().conj().T
                            @ self.previous_grad.ravel()
                        )
                    ),
                )
                self.previous_grad = grad
                self.previous_direction = (
                    -grad + beta * self.previous_direction
                )
                return self.previous_direction


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
    nabla_f,
    x0,
    n_iter,
    tau,
    callback=lambda x: None,
):
    x = x0.clone()
    for _ in range(n_iter):
        x -= tau * nabla_f(x)
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

    for _ in range(n_iter):
        J = f.jacobian(x)
        r = f(x)
        dx = solve(J.T @ J, J.T @ r)
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
        return (a * b).sum(dim=dim)

    x = x0.clone()
    r = b - A @ x
    p = r
    for _ in range(n_iter):
        Ap = A @ p
        alpha = inner(r.conj(), r) / inner(p, Ap)
        x = x + alpha * p
        r_new = r - alpha * Ap
        beta = inner(r_new.conj(), r_new) / inner(r.conj(), r)
        p = r_new + beta * p
        r = r_new.clone()
        callback(x)

        # TODO see above; adapt stopping criterion to batched
        # dont re-implement the bug we had with mumy
        if (th.sqrt((r**2).sum(dim=dim)) < tol).all():
            break
    return x


class GerchbergSaxton:
    def __init__(self, near_field_intensity, far_field_intensity):
        self.amp_fourier_space = th.sqrt(far_field_intensity)
        self.amp_real_space = th.sqrt(near_field_intensity)

        self.x_shape = near_field_intensity.shape
        self.lost_list = []

    def iterate(self, initial_est=None, n_iter=100):
        if initial_est is not None:
            phase_est = th.copy(initial_est)
        else:
            phase_est = th.ones(shape=self.x_shape, dtype=th.complex128)

        field_real_space = self.amp_real_space * th.exp(1j * phase_est)
        for i_iter in range(n_iter):
            _field_fourier_space = th.fft.fft2(field_real_space)
            field_fourier_space = self.amp_fourier_space * th.exp(
                1j * th.angle(_field_fourier_space)
            )

            _field_real_space = th.fft.ifft2(field_fourier_space)
            field_real_space = self.amp_real_space * th.exp(
                1j * th.angle(_field_real_space)
            )

            lost = th.sum(
                (th.abs(_field_fourier_space) - self.amp_fourier_space) ** 2
            ) / th.sum((self.amp_fourier_space) ** 2)
            self.lost_list.append(lost)

        return field_real_space
