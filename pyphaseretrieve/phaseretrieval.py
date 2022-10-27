import numpy as np
from pyphaseretrieve.linop import *

class PhaseRetrievalBase:
    def __init__(self, linop:BaseLinOp):
        self.linop = linop

    def apply(self, x):
        return self.linop.apply(x)
    
    def applyAdjoint(self, x):
        return self.linop.applyAdjoint(x)

class Ptychography1d(PhaseRetrievalBase):
    def __init__(self, probe, shifts=None, n_img=10):
        self.probe = probe
        self.probe_size = probe.shape[0]

        if shifts is not None:
            self.shifts = shifts
            self.n_img = len(shifts)
        else:
            self.n_img = n_img
            self.probe_dia = np.count_nonzero(self.probe)
            self.upper_origin = self.probe_dia / 2
            self.lower_origin = self.probe_size - self.probe_dia / 2
            self.step_size = int((self.lower_origin - self.upper_origin) / self.n_img)
            
            self.shifts = []
            for _idx in range(self.n_img):
                self.shifts.append(_idx*self.step_size)
        
        op_fft = LinOpFFT()
        op_probe = LinOpMul(self.probe)
        self.linop = StackLinOp([
            op_fft @ op_probe @ LinOpRoll(self.shifts[i_probe], 0)
            for i_probe in range(self.n_img)
        ])
        self.in_size = self.linop.in_size

    def spectralinit(self, y, n_iter=100, method="Lu"):
        in_size = self.linop.in_size
        
        y_ = y/np.mean(y)  
        if method == "Lu":  
            t = np.maximum(1-1/y_, np.array(-1)) 
        else:  
            t = y_

        x_est = np.random.randn(in_size,)
        for i_iter in range(np.minimum(n_iter, 10)):
            x_new = self.linop.apply(x_est)
            x_new = t * x_new
            x_new = self.linop.applyAdjoint(x_new)
            x_est = x_new / np.linalg.norm(x_new)

        x_new = self.linop.apply(x_est)
        x_new = t * x_new
        x_new = self.linop.applyAdjoint(x_new)
        corr = np.real(x_new.T.conj() @ x_est)
        
        if corr < 0:  
            for i_iter in range(n_iter):
                x_new = self.linop.apply(x_est)
                x_new = t * x_new
                x_new = self.linop.applyAdjoint(x_new)
                x_new = x_new + 1.1*np.abs(corr)*x_est
                x_est = x_new / np.linalg.norm(x_new)
        else: 
            for i_iter in range(n_iter - 10):
                x_new = self.linop.apply(x_est)
                x_new = t * x_new
                x_new = self.linop.applyAdjoint(x_new)
                x_est = x_new / np.linalg.norm(x_new)
        return x_est

    def overlap_rate(self) -> float:
        overlap = 1 - self.step_size / self.probe_dia
        return overlap

    
class Ptychography2d(PhaseRetrievalBase):
    def __init__(self, probe, shifts_pair = None, reconstruct_size = None, n_img:int = 16):
        # assume reconstruction size is square
        # shifts_pair = [v_shifts, h_shifts]
        self.probe            = probe
        self.probe_size       = probe.shape[0]

        idx = 0
        while self.probe[idx][0] != 0:
            idx += 1
        self.probe_dia = (idx-1)*2

        if reconstruct_size is not None:
            self.reconstruct_size = reconstruct_size
        else:
            self.reconstruct_size = self.probe_size

        if shifts_pair is not None:
            assert shifts_pair.shape[1] == 2 , "shifts_map dimension should be (n,2)"
            self.shifts_pair = shifts_pair
            self.n_img       = shifts_pair.shape[0]

        else:
            pass
            # assert int(math.sqrt(n_img))**2 == n_img, "n_img need to be perfect square"
            # self.n_img = n_img

            # left_origin = self.probe_dia / 2
            # right_origin = self.probe_size - self.probe_dia / 2
            # step_size = int((right_origin - left_origin) / int(math.sqrt(self.n_img)))

            # shifts_h = []
            # shifts_v = []
            # for _idx in range(int(math.sqrt(self.n_img))):
            #     shifts_h.append(_idx*step_size)
            #     shifts_v.append(_idx*step_size)

            # shifts_h = np.array(shifts_h)
            # shifts_v = np.array(shifts_v)

            # self.shifts_pair = np.concatenate([shifts_h.reshape(self.n_img,1),shifts_v.reshape(self.n_img,1)],axis=1)
            # print(self.shifts_pair)
           
        op_fft2     = LinOpFFT2()
        op_ifft2    = LinOpIFFT2() 
        op_crop     = LinOpCrop2D(self.reconstruct_size ,self.probe_size)
        op_probe    = LinOpMul(self.probe)
        self.linop = Stack2DLinOp([
            op_ifft2 @ op_probe @ op_crop @ LinOpRoll(int(self.shifts_pair[i_probe,1]), 1) @ LinOpRoll(int(self.shifts_pair[i_probe,0]), 0) @ op_fft2
            for i_probe in range(self.n_img)
        ])
        self.in_size = self.linop.in_size
    
    def overlap_rate(self):
        pass