import numpy as np
from typing import Optional
import torch as th
import pyphaseretrieve.loss as pl
import pyphaseretrieve.phaseretrieval as pp
import pyphaseretrieve.linop as plinop


class GradientDescent:
    def __init__(
        self,
        pr_model: pp.PhaseRetrievalBase,
        loss_func: Optional[pl.LossFunction] = None,
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
        self.current_iter = 0
        self.loss_list = []

    def iterate(self, y, initial_est=None, n_iter=100, lr=1, alpha=0):
        if initial_est is not None:
            x_est = np.copy(initial_est)
        else:
            x_est = np.ones(shape=self.x_shape, dtype=np.complex128)

        for i_iter in range(n_iter):
            loss, grad = self.loss_func.compute_loss(y, self.pr_model, x_est)
            self.loss_list.append(loss)
            self.current_iter += 1

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
            x_est = np.copy(initial_est)
        else:
            x_est = np.ones(shape=self.x_shape, dtype=np.complex128)

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
                    PhaseRetrievalBase(i_linop),
                    x_est,
                )
                x_est += lr * (-grad)
                current_idx += self.pr_model.probe_shape[0]
            self.current_iter += 1
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
        m = np.real(current_grad.ravel().T.conj() @ descent_direction.ravel())
        if m >= 0:
            print(
                "There may be a sign error in the computation of the descent direction."
            )
        while True:
            new_x_est = x_est + lr * descent_direction
            new_y_est = np.abs(self.pr_model.apply(new_x_est)) ** 2
            new_loss = np.sum((new_y_est - y) ** 2)
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
                beta = np.maximum(
                    0,
                    np.real(
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
    def __init__(self, pr_model: pp.PhaseRetrievalBase):
        self.pr_model = pr_model
        self.x_shape = pr_model.in_shape

    def iterate(self, y, initial_est=None, n_iter=100, method="Lu"):
        y_norm = y / np.mean(y)
        if method == "Lu":
            threshold = np.maximum(1 - 1 / y_norm, -1)
        else:
            threshold = y_norm

        if initial_est is not None:
            x_est = np.copy(initial_est)
        else:
            x_est = np.random.randn(*self.x_shape)

        for i_iter in range(np.minimum(n_iter, 10)):
            x_new = self.pr_model.apply(x_est)
            x_new = threshold * x_new
            x_new = self.pr_model.applyT(x_new)
            x_est = x_new / np.linalg.norm(x_new)

        x_new = self.pr_model.apply(x_est)
        x_new = threshold * x_new
        x_new = self.pr_model.applyT(x_new)
        corr = np.real(x_new.ravel().T.conj() @ x_est.ravel())

        if corr < 0:
            for i_iter in range(n_iter):
                x_new = self.pr_model.apply(x_est)
                x_new = threshold * x_new
                x_new = self.pr_model.applyT(x_new)
                x_new = x_new + 1.1 * np.abs(corr) * x_est
                x_est = x_new / np.linalg.norm(x_new)
        else:
            for i_iter in range(n_iter - 10):
                x_new = self.pr_model.apply(x_est)
                x_new = threshold * x_new
                x_new = self.pr_model.applyT(x_new)
                x_est = x_new / np.linalg.norm(x_new)
        return x_est


class PerturbativePhase:
    def __init__(
        self,
        pr_model: pp.PhaseRetrievalBase,
        loss_func: Optional[pl.LossFunction] = None,
    ):
        """min {Y-|Ax|**2 - B*epsilon}"""
        self.pr_model = pr_model

        if loss_func is None:
            self.loss_func = pl.loss_intensity_based()
        else:
            self.loss_func = loss_func

        self.x_shape = pr_model.in_shape
        self.current_iter = 0
        self.loss_list = []

    def iterate_GradientDescent(
        self,
        y,
        initial_est=None,
        n_iter=100,
        linear_n_iter=20,
        lr=1e-1,
        alpha=0,
    ):
        if initial_est is not None:
            x_est = np.copy(initial_est)
        else:
            x_est = np.ones(shape=self.x_shape, dtype=np.complex128)

        for i_iter in range(n_iter):
            loss = self.loss_func.compute_loss(
                y, self.pr_model, x_est, compute_grad=False
            )
            self.loss_list.append(loss)
            self.current_iter += 1

            perturbative_model, _ = self.pr_model.get_perturbative_model(
                x_est, method="GradientDescent"
            )

            y_est = self.pr_model.apply_ModularSquare(x_est)
            epsilon = np.zeros_like(x_est)
            for _ in range(linear_n_iter):
                grad = -2 * perturbative_model.applyT(
                    y - y_est - perturbative_model.apply(epsilon)
                )
                grad = grad + alpha * x_est
                epsilon = epsilon - lr * grad
            print(self.current_iter)
            x_est += epsilon
        return x_est

    # TODO this is Gauss-Newton?
    def iterate_ConjugateGradientDescent(
        self,
        y,
        x0=None,
        n_iter=100,
        linear_n_iter=20,
        tolerance=1e-9,
        _lambda=0,
    ):
        x = x0.copy()

        for i_iter in range(n_iter):
            loss = self.loss_func.compute_loss(
                y, self.pr_model, x, compute_grad=False
            )
            self.loss_list.append(loss)
            self.current_iter += 1

            perturbative_model, _ = self.pr_model.get_perturbative_model(
                x, method="ConjugateGradientDescent"
            )

            y_est = self.pr_model.apply_ModularSquare(x)
            A = perturbative_model.T() @ perturbative_model
            epsilon = cg(
                A,
                perturbative_model.applyT(y - y_est),
                np.zeros((*x.shape, 2)),
                n_iter=linear_n_iter,
                tol=tolerance,
            )
            x += epsilon[..., 0] + epsilon[..., 1] * 1j

        return x


# TODO implement batched
def cg(
    A: plinop.BaseLinOp,
    b: th.Tensor,
    x0: th.Tensor,
    n_iter: int = 100,
    tol: float = 1e-9,
    dim: tuple[int, ...] = (0, 1, 2),
):
    x = x0.copy()
    r = b - A.apply(x)
    p = r
    for _ in range(n_iter):
        Ap = A.apply(p)
        alpha = (r.conj() * r).sum(axis=dim) / (p * Ap).sum()
        x = x + alpha * p
        r_new = r - alpha * Ap
        beta = (r_new.conj() * r_new).sum(axis=dim) / (r.conj() * r).sum(
            axis=dim
        )
        p = r_new + beta * p
        r = r_new.copy()
        if np.sqrt((r**2).sum(axis=dim)) < tol:
            break
    return x


class GerchbergSaxton:
    def __init__(self, near_field_intensity, far_field_intensity):
        self.amp_fourier_space = np.sqrt(far_field_intensity)
        self.amp_real_space = np.sqrt(near_field_intensity)

        self.x_shape = near_field_intensity.shape
        self.lost_list = []
        self.current_iter = 0

    def iterate(self, initial_est=None, n_iter=100):
        if initial_est is not None:
            phase_est = np.copy(initial_est)
        else:
            phase_est = np.ones(shape=self.x_shape, dtype=np.complex128)

        field_real_space = self.amp_real_space * np.exp(1j * phase_est)
        for i_iter in range(n_iter):
            _field_fourier_space = np.fft.fft2(field_real_space)
            field_fourier_space = self.amp_fourier_space * np.exp(
                1j * np.angle(_field_fourier_space)
            )

            _field_real_space = np.fft.ifft2(field_fourier_space)
            field_real_space = self.amp_real_space * np.exp(
                1j * np.angle(_field_real_space)
            )

            lost = np.sum(
                (np.abs(_field_fourier_space) - self.amp_fourier_space) ** 2
            ) / np.sum((self.amp_fourier_space) ** 2)
            self.lost_list.append(lost)
            self.current_iter += 1

        return field_real_space
