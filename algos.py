import numpy as np
from linop import *
from phaseretrieval import *

class GradientDescent:
    def __init__(self, pr_model, line_search=None, acceleration=None):
        self.pr_model = pr_model
        self.line_search = line_search
        self.acceleration = acceleration

        self.x_shape = pr_model.in_size
        self.current_iter = 0
        self.loss_list = []

    def iterate(self, y, initial_est=None, 
                n_iter=100, lr=1e-1):
        if initial_est is not None:
            x_est = initial_est
        else:
            x_est = np.ones(shape=self.x_shape, dtype=np.complex64)

        for i_iter in range(n_iter):
            loss, grad = _compute_loss_gradient(y, self.pr_model, x_est)
            self.loss_list.append(loss)
            self.current_iter += 1

            if self.acceleration is not None:
                descent_direction = self.find_dir(x_est, y, grad, self.acceleration)
            else:
                descent_direction = -grad

            if self.line_search is not None:
                actual_lr = self.find_lr(x_est, y, descent_direction, grad, loss, initial_lr=lr)
            else:
                actual_lr = lr

            x_est -= actual_lr * grad
        
        return x_est

    def find_lr(self, x_est, y, descent_direction, 
                    current_grad, initial_loss, initial_lr=1, c=0.9, tau=0.5):
        lr = initial_lr
        m = np.real(current_grad.ravel().T.conj() @ descent_direction.ravel())
        if m >= 0:
            print("There may be a sign error in the computation of the descent direction.")
        new_x_est = x_est + lr * descent_direction
        new_y_est = self.pr_model.apply(new_x_est)
        new_loss = np.sum((new_y_est-y)**2)
        while new_loss > initial_loss + lr * c * m:
            lr = tau * lr
            new_x_est = x_est + lr * descent_direction
            new_y_est = self.pr_model.apply(new_x_est)
            new_loss = np.sum((new_y_est-y)**2)
        return lr

    def find_dir(self, x_est, y, grad):
        if self.acceleration == "conjugate gradient":
            if not hasattr(self, "previous_grad"):
                self.previous_grad = grad
                self.previous_direction = -grad
                return -grad
            else:
                beta = np.maximum(0, np.real(grad.ravel().conj().T @ 
                    (grad.ravel() - self.previous_grad.ravel()) / 
                    (self.previous_grad.ravel().conj().T @ self.previous_grad.ravel()))) #TOCHECK
                self.previous_grad = grad
                self.previous_direction = -grad + beta * self.previous_direction
                return self.previous_direction
        


def _compute_loss_gradient(y, pr_model, x_est):
    out_field = pr_model.linop.apply(x_est)
    est_y = np.abs(out_field)**2
    loss = np.sum((est_y - y)**2)
    grad = pr_model.linop.applyAdjoint( (est_y - y) * out_field )

    return loss, grad
    


