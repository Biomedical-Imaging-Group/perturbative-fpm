import numpy as np
from abc import abstractmethod
from pyphaseretrieve.phaseretrieval import*
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

    def compute_loss(self, y, pr_model:PhaseRetrievalBase, x_est, compute_grad:bool= True):
        y_est = pr_model.apply_ModularSquare(x_est)
        loss = np.sum((y_est - y)**2)  
        if compute_grad:
            out_field = pr_model.apply(x_est)
            grad = -2 * pr_model.applyT( out_field * (y - y_est) )
            return loss, grad
        else:
            return loss

class loss_amplitude_based(LossFunction):
    def __init__(self, epsilon= 0):
        self.epsilon = epsilon

    def compute_loss(self, y, pr_model:PhaseRetrievalBase, x_est, compute_grad:bool= True):
        y_est = pr_model.apply_ModularSquare(x_est)
        loss = np.sum((np.sqrt(y_est) - np.sqrt(y))**2)  
        if compute_grad:
            out_field = pr_model.apply(x_est)
            grad = pr_model.applyT( out_field - out_field/(np.sqrt(y_est)+self.epsilon)*np.sqrt(y) )
            return loss, grad
        else:
            return loss