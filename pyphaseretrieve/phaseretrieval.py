import numpy as np
from linop import *

class PhaseRetrievalBase:
    def __init__(self, linop):
        self.linop = linop

    def apply(self, x):
        return np.abs(self.linop.apply(x))**2

class Ptychography1d(PhaseRetrievalBase):
    def __init__(self, probe, shifts=None, n_img=10):
        self.probe = probe
        self.probe_size = probe.shape[0]
        if shifts is not None:
            self.shifts = shifts
            self.n_img = len(shifts)
        else:
            self.n_img = n_img
            self.shifts = np.round(
                np.arange(0, self.probe_size-1, step=self.probe_size/n_img)).astype(int)
        
        print(self.shifts)
        op_fft = LinOpFFT()
        op_probe = LinOpMul(self.probe)
        self.linop = StackLinOp([
            op_fft @ op_probe @ LinOpRoll(self.shifts[i_probe], 0)
            for i_probe in range(self.n_img)
        ])
        # self.linop = op_fft @ op_probe @ LinOpRoll(self.shifts[0], 0)
        # for i_probe in range(1, self.n_img):
        #     self.linop = StackLinOp([self.linop, 
        #         op_fft @ op_probe @ LinOpRoll(self.shifts[i_probe], 0)])
        
