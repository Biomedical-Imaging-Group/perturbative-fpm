import numpy as np
from abc import abstractmethod
from pyphaseretrieve.linop import*

class LossFunction:
    def __init__(self):
        pass

    @abstractmethod
    def compute_loss(self):
        pass

class loss_intensity_based(LossFunction):
    def __init__(self):
        pass

    def compute_loss(self, y, pr_model, x_est):
        out_field = pr_model.apply(x_est)
        est_y = np.abs(out_field)**2
        loss = np.sum((est_y - y)**2)  
        grad = 2 * pr_model.applyT( (est_y - y) * out_field )

        return loss, grad

class loss_amplitude_based(LossFunction):
    def __init__(self, epsilon= 0):
        self.epsilon = epsilon

    def compute_loss(self, y, pr_model, x_est):
        out_field = pr_model.apply(x_est)
        est_y = np.abs(out_field)**2
        loss = np.sum((np.sqrt(est_y) - np.sqrt(y))**2)  
        grad = pr_model.applyT( out_field - out_field/(np.sqrt(est_y)+self.epsilon)*np.sqrt(y) )

        return loss, grad

class loss_perturbative_based(LossFunction):
    def __init__(self, n_iter= 50, lr= 1e-7):
        self.n_iter = n_iter
        self.lr = lr

    def compute_loss(self, y, pr_model, x_est):
        out_field = pr_model.apply(x_est)
        est_y = np.abs(out_field)**2
        loss = np.sum((est_y - y)**2) 

        perturbative_model = 2 * LinOpReal() @ LinOpMul(out_field.conj()) @ pr_model.linop
        epsilon = np.random.randn(*pr_model.linop.in_shape)
        for i_iter in range(self.n_iter):
            grad = -2 * perturbative_model.applyT(y - est_y - perturbative_model.apply(epsilon))
            epsilon = epsilon - self.lr*grad 

        return loss, epsilon