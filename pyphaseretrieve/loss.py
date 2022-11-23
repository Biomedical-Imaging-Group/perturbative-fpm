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

    def compute_loss(self, y, pr_model, x_est, compute_grad:bool = True):
        out_field = pr_model.apply(x_est)
        y_est = np.abs(out_field)**2
        loss = np.sum((y_est - y)**2)  
        if compute_grad:
            grad = 2 * pr_model.applyT( (y_est - y) * out_field )
            return loss, grad
        else:
            return loss

class loss_amplitude_based(LossFunction):
    def __init__(self, epsilon= 0):
        self.epsilon = epsilon

    def compute_loss(self, y, pr_model, x_est, compute_grad:bool = True):
        out_field = pr_model.apply(x_est)
        y_est = np.abs(out_field)**2
        loss = np.sum((np.sqrt(y_est) - np.sqrt(y))**2)  
        if compute_grad:
            grad = pr_model.applyT( out_field - out_field/(np.sqrt(y_est)+self.epsilon)*np.sqrt(y) )
            return loss, grad
        else:
            return loss