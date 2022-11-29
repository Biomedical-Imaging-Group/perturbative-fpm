import numpy as np
import cupy as cp
from pyphaseretrieve.loss       import *
from pyphaseretrieve.phaseretrieval import *


class GradientDescent:
    def __init__(self, pr_model:PhaseRetrievalBase, loss_func:LossFunction=None, line_search=None, acceleration=None, GPU_device:bool= False):
        self.pr_model = pr_model
        if loss_func is not None:
            self.loss_func = loss_func
        else:
            self.loss_func = loss_intensity_based()

        self.line_search = line_search
        self.acceleration = acceleration

        self.x_shape = pr_model.linop.in_shape
        self.current_iter = 0
        self.loss_list = []

        self.GPU_device = GPU_device

    def iterate(self, y, initial_est=None, n_iter=100, lr=1) -> np.ndarray:
        if initial_est is not None:
            x_est = np.copy(initial_est)
        else:
            x_est = np.ones(shape= self.x_shape, dtype= np.complex128)

        if self.GPU_device:
            x_est = cp.array(x_est)
            y = cp.array(y)

        for i_iter in range(n_iter):
            loss, grad = self.loss_func.compute_loss(y, self.pr_model, x_est)
            self.loss_list.append(loss)
            self.current_iter += 1

            if self.acceleration is not None:
                descent_direction = self.find_dir(x_est, y, grad)
            else:
                descent_direction = -grad

            if self.line_search is not None:
                actual_lr = self.find_lr(x_est, y, descent_direction, grad, loss, initial_lr=lr)
            else:
                actual_lr = lr

            x_est += actual_lr * descent_direction
        return x_est if isinstance(x_est, np.ndarray) else np.array(x_est.get())

    def find_lr(self, x_est, y, descent_direction, 
                    current_grad, initial_loss, initial_lr=1, c=0.9, tau=0.5):
        lr = initial_lr
        m = np.real(current_grad.ravel().T.conj() @ descent_direction.ravel())
        if m >= 0:
            print("There may be a sign error in the computation of the descent direction.")
        while True:
            new_x_est = x_est + lr * descent_direction
            new_y_est = np.abs(self.pr_model.apply(new_x_est))**2
            new_loss = np.sum((new_y_est-y)**2)
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
                beta = np.maximum(0, np.real(grad.ravel().conj().T @ 
                    (grad.ravel() - self.previous_grad.ravel()) / 
                    (self.previous_grad.ravel().conj().T @ self.previous_grad.ravel()))) #TOCHECK
                self.previous_grad = grad
                self.previous_direction = -grad + beta * self.previous_direction
                return self.previous_direction

class SpectralMethod:  ## only validate in 1D!!
    def __init__(self, pr_model:PhaseRetrievalBase):
        self.pr_model = pr_model
        self.x_shape = pr_model.linop.in_shape

    def iterate(self, y, n_iter=100, method="Lu"):       
        y_norm = y/np.mean(y)  
        if method == "Lu":  
            threshold = np.maximum(1-1/y_norm, np.array(-1)) 
        else:  
            threshold = y_norm

        x_est = np.random.randn(*self.x_shape)
        for i_iter in range(np.minimum(n_iter, 10)):
            x_new = self.pr_model.linop.apply(x_est)
            x_new = threshold * x_new
            x_new = self.pr_model.linop.applyT(x_new)
            x_est = x_new / np.linalg.norm(x_new)

        x_new = self.pr_model.linop.apply(x_est)
        x_new = threshold * x_new
        x_new = self.pr_model.linop.applyT(x_new)
        corr = np.real(x_new.T.conj() @ x_est)
        
        if corr < 0:  
            for i_iter in range(n_iter):
                x_new = self.pr_model.linop.apply(x_est)
                x_new = threshold * x_new
                x_new = self.pr_model.linop.applyT(x_new)
                x_new = x_new + 1.1*np.abs(corr)*x_est
                x_est = x_new / np.linalg.norm(x_new)
        else: 
            for i_iter in range(n_iter - 10):
                x_new = self.pr_model.linop.apply(x_est)
                x_new = threshold * x_new
                x_new = self.pr_model.linop.applyT(x_new)
                x_est = x_new / np.linalg.norm(x_new)
        return x_est
    
class PerturbativePhase:
    def __init__(self, pr_model:PhaseRetrievalBase, loss_func:LossFunction=None):
        """min {Y-|Ax|**2 - B*epsilon}"""
        self.pr_model = pr_model
        if loss_func is not None:
            self.loss_func = loss_func
        else:
            self.loss_func = loss_intensity_based()

        self.x_shape = pr_model.linop.in_shape
        self.current_iter = 0
        self.loss_list = []

    def iterate_GD(self, y, initial_est=None, n_iter=100, GD_n_iter=20, lr=1e-1):
        if initial_est is not None:
            x_est = np.copy(initial_est)
        else:
            x_est = np.ones(shape= self.x_shape, dtype= np.complex128)

        for i_iter in range(n_iter):
            loss = self.loss_func.compute_loss(y, self.pr_model, x_est, compute_grad= False)
            self.loss_list.append(loss)
            self.current_iter += 1

            out_field = self.pr_model.apply(x_est)
            y_est = np.abs(out_field)**2

            perturbative_model = 2 * LinOpReal() @ LinOpMul(out_field.conj()) @ self.pr_model.linop
            
            epsilon = np.zeros(self.pr_model.linop.in_shape).astype(np.complex128)
            for gd_i_iter in range(GD_n_iter):
                grad = (-2 * perturbative_model.applyT(y - y_est - perturbative_model.apply(epsilon)))                   
                epsilon = epsilon - lr*grad
            print(self.current_iter)

            x_est += epsilon
        return x_est

    def iterate_CGD(self, y, initial_est=None, n_iter=100, CGD_n_iter=20):
        if initial_est is not None:
            x_est = np.copy(initial_est)
        else:
            x_est = np.ones(shape= self.x_shape, dtype= np.complex128)

        for i_iter in range(n_iter):
            loss = self.loss_func.compute_loss(y, self.pr_model, x_est, compute_grad= False)
            self.loss_list.append(loss)
            self.current_iter += 1

            out_field = self.pr_model.apply(x_est)
            y_est = np.abs(out_field)**2

            perturbative_model = RealPartExpandOp (2 * LinOpMul(out_field.conj()) @ self.pr_model.linop)
            
            expand_shape = np.array(self.pr_model.linop.in_shape)
            expand_shape[0] = expand_shape[0]*2
            epsilon_expand = np.zeros(tuple(expand_shape)).astype(np.complex128)
            res = (perturbative_model.applyT(perturbative_model.apply(epsilon_expand))) - perturbative_model.applyT(y - y_est)
            descent_direction = -res
            for cgd_i_iter in range(CGD_n_iter):
                alpha = (res.ravel().T.conj() @ res.ravel())/(descent_direction.ravel().T.conj() @ perturbative_model.applyT(perturbative_model.apply(descent_direction)).ravel())
                epsilon_expand = epsilon_expand + alpha*descent_direction
                r_new = res + alpha * (perturbative_model.applyT(perturbative_model.apply(descent_direction)))
                descent_direction = -r_new + ((r_new.ravel().T.conj()@r_new.ravel())/(res.ravel().T.conj()@res.ravel()))*descent_direction
                res = r_new
            epsilon = epsilon_expand[:self.pr_model.linop.in_shape[0]] + epsilon_expand[self.pr_model.linop.in_shape[0]:]* 1j

            x_est += epsilon
        return x_est

class GerchbergSaxton: 
    def __init__(self, near_field_intensity, far_field_intensity):
        self.amp_fourier_space = np.sqrt(far_field_intensity)   
        self.amp_real_space = np.sqrt(near_field_intensity)

        self.x_shape = near_field_intensity.shape
        self.lost_list = []
        self.current_iter = 0

    def iterate(self, initial_est= None, n_iter= 100):
        if initial_est is not None:
            phase_est = np.copy(initial_est)
        else:
            phase_est = np.ones(shape= self.x_shape, dtype= np.complex128)

        field_real_space = self.amp_real_space * np.exp(1j*phase_est)
        for i_iter in range(n_iter):
            _field_fourier_space = np.fft.fft2(field_real_space)
            field_fourier_space = self.amp_fourier_space * np.exp(1j*np.angle(_field_fourier_space))

            _field_real_space = np.fft.ifft2(field_fourier_space)
            field_real_space = self.amp_real_space * np.exp(1j*np.angle(_field_real_space))

            lost = np.sum((np.abs(_field_fourier_space) - self.amp_fourier_space)**2) / np.sum((self.amp_fourier_space)**2)
            self.lost_list.append(lost)
            self.current_iter += 1
        
        return field_real_space