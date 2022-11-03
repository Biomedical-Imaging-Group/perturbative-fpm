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
        self.in_size = probe.shape[0]

        if shifts is not None:
            self.n_img = len(shifts) 
            self.shifts = shifts
        else:
            self.n_img = n_img
            self.probe_dia = np.count_nonzero(self.probe)
            self.step_size = int((self.probe_size - (self.probe_dia/2)) / self.n_img)
            
            self.shifts = []
            for _idx in range(self.n_img):
                self.shifts.append(_idx * self.step_size)
        
        op_fft = LinOpFFT()
        op_probe = LinOpMul(self.probe)
        self.linop = StackLinOp([
            op_fft @ op_probe @ LinOpRoll(self.shifts[i_probe], 0)
            for i_probe in range(self.n_img)
        ])

    def spectralinit(self, y, n_iter=100, method="Lu"):       
        y_norm = y/np.mean(y)  
        if method == "Lu":  
            threshold = np.maximum(1-1/y_norm, np.array(-1)) 
        else:  
            threshold = y_norm

        x_est = np.random.randn(self.in_size,)
        for i_iter in range(np.minimum(n_iter, 10)):
            x_new = self.linop.apply(x_est)
            x_new = threshold * x_new
            x_new = self.linop.applyAdjoint(x_new)
            x_est = x_new / np.linalg.norm(x_new)

        x_new = self.linop.apply(x_est)
        x_new = threshold * x_new
        x_new = self.linop.applyAdjoint(x_new)
        corr = np.real(x_new.T.conj() @ x_est)
        
        if corr < 0:  
            for i_iter in range(n_iter):
                x_new = self.linop.apply(x_est)
                x_new = threshold * x_new
                x_new = self.linop.applyAdjoint(x_new)
                x_new = x_new + 1.1*np.abs(corr)*x_est
                x_est = x_new / np.linalg.norm(x_new)
        else: 
            for i_iter in range(n_iter - 10):
                x_new = self.linop.apply(x_est)
                x_new = threshold * x_new
                x_new = self.linop.applyAdjoint(x_new)
                x_est = x_new / np.linalg.norm(x_new)
        return x_est

    def overlap_rate(self) -> float:
        overlap = 1 - self.step_size / self.probe_dia
        return overlap

    
class FourierPtychography2d(PhaseRetrievalBase):
    def __init__(self, probe, shifts_pair= None, reconstruct_size= None, n_img:int= 25):
        """shifts_pair is defined as [v_shifts,h_shifts]"""
        self.probe = probe
        self.probe_size = probe.shape[0]

        if reconstruct_size is not None:
            self.reconstruct_size = reconstruct_size
        else:
            self.reconstruct_size = self.probe_size
        self.in_size = self.reconstruct_size

        if shifts_pair is not None:
            assert shifts_pair.shape[1] == 2 , "shifts_map dimension should be (n,2)"
            self.shifts_pair = shifts_pair
            self.n_img = shifts_pair.shape[0]
        else:
            assert int(np.sqrt(n_img))**2 == n_img, "n_img need to be perfect square"
            self.n_img = n_img
            side_n_img = np.sqrt(self.n_img)

            shift_probe = np.fft.fftshift(self.probe)
            probe_center_row = shift_probe[int(self.probe_size//2)]    
            self.probe_diameter = np.count_nonzero(probe_center_row)

            self.step_size = int((self.reconstruct_size - 2*int(self.probe_diameter//2)) // (side_n_img-1))

            shifts_h = []
            shifts_v = []
            for _idx in np.arange(-np.floor(side_n_img/2), np.ceil(side_n_img/2), 1):
                shifts_h.append(_idx * self.step_size)
                shifts_v.append(_idx * self.step_size)
            shifts_h, shifts_v = np.meshgrid(shifts_h, shifts_v)

            self.shifts_pair = np.concatenate([shifts_v.reshape(self.n_img,1), shifts_h.reshape(self.n_img,1)], axis=1)

        op_ifft2 = LinOpIFFT2() 
        op_fcrop = LinOpFourierCrop2d(self.reconstruct_size, self.probe_size)
        op_probe = LinOpMul(self.probe)
        self.linop = StackLinOp([
            op_ifft2 @ op_probe @ op_fcrop @ LinOpRoll(int(self.shifts_pair[i_probe,1]), 1) @ LinOpRoll(int(self.shifts_pair[i_probe,0]), 0)
            for i_probe in range(self.n_img)
        ])

    def get_probe_map(self):
        pad_size    = self.reconstruct_size - self.probe_size
        shift_probe = np.fft.fftshift(self.probe)
        shift_probe = np.pad(shift_probe ,(int(np.floor(pad_size/2)), int(np.ceil(pad_size/2))), mode='constant')

        overlap_img = np.zeros(shape= (self.reconstruct_size, self.reconstruct_size))
        for i_probe in range(self.n_img):
            roll_linop  = LinOpRoll(int(self.shifts_pair[i_probe,1]), 1) @ LinOpRoll(int(self.shifts_pair[i_probe,0]),0)
            overlap_img = overlap_img + roll_linop.apply(shift_probe)

        return overlap_img
    
    def overlap_rate(self):
        if self.probe_diameter is None:
            return None
        else:
            probe_radius = self.probe_diameter//2
            if self.step_size > (probe_radius*2):
                return 0
            else:
                circ_sector     = 2*(np.arccos(self.step_size/2/probe_radius)/(2*np.pi)) * np.pi*probe_radius**2
                tria_area       = self.step_size/2 * np.sqrt(probe_radius**2 - (self.step_size/2)**2)
                overlap_rate    = 2*(circ_sector - tria_area)/np.pi/(probe_radius**2)
                return overlap_rate