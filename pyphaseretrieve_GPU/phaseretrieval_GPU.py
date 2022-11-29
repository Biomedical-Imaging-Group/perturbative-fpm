from abc import abstractmethod
import numpy as np
import cupy as cp
from pyphaseretrieve_GPU.linop_GPU import*

class PhaseRetrievalBase:
    def __init__(self, linop:BaseLinOp):
        self.linop = linop

    def apply(self, x):
        return self.linop.apply(x)
    
    def applyT(self, x):
        return self.linop.applyT(x)

class Ptychography1d(PhaseRetrievalBase):
    def __init__(self, probe:np.ndarray, shifts:np.ndarray=None, n_img:int=10):
        self.probe = probe
        self.probe_shape = probe.shape

        if shifts is not None:
            self.n_img = len(shifts) 
            self.shifts = shifts
        else:
            self.n_img = n_img
            self.shifts = self.get_auto_shifts()

        self.linop = self.get_forward_model()

    @abstractmethod
    def get_auto_shifts(self) -> np.ndarray:
        probe_dia = np.count_nonzero(self.probe)
        start_shift = -(self.probe_shape[0]-probe_dia)//2
        end_shift = (self.probe_shape[0]-probe_dia)//2
        shifts = np.linspace(start_shift, end_shift, self.n_img)
        return shifts

    def get_forward_model(self) -> BaseLinOp:
        probe = cp.array(self.probe)
        op_fft = LinOpFFT()
        op_probe = LinOpMul(probe)
        linop = StackLinOp([
            op_fft @ op_probe @ LinOpRoll(self.shifts[i_probe])
            for i_probe in range(self.n_img)
        ])
        return linop
    
    def get_probe_overlap_array(self) -> np.ndarray:
        overlap_img = np.zeros(shape= self.probe_shape)
        for i_probe in range(self.n_img):
            roll_linop  = LinOpRoll(self.shifts[i_probe])
            overlap_img = overlap_img + roll_linop.apply(self.probe)
        return overlap_img

    def overlap_rate(self) -> float:
        probe_dia = np.count_nonzero(self.probe)
        step_size = np.abs(self.shifts[0]-self.shifts[1])
        overlap = 1 - step_size / probe_dia
        return overlap

class FourierPtychography2d(PhaseRetrievalBase):
    def __init__(self, probe:np.ndarray, shifts_pair:np.ndarray= None, reconstruct_size:int= None, n_img:int= 25):
        """shifts_pair is defined as [v_shifts,h_shifts]"""
        self.probe = probe
        self.probe_shape = probe.shape

        if reconstruct_size is not None:
            self.reconstruct_size = reconstruct_size
        else:
            self.reconstruct_size = self.probe_shape[0]

        if shifts_pair is not None:
            assert shifts_pair.ndim == 2 , "shifts_map dimension should be (n,2)"
            self.n_img = shifts_pair.shape[0]
            self.shifts_pair = shifts_pair
        else:
            assert int(np.sqrt(n_img))**2 == n_img, "n_img need to be perfect square"
            self.n_img = n_img
            self.shifts_pair = self.get_auto_shifts_pair()
            
        self.linop = self.get_forward_model()

    @abstractmethod
    def get_auto_shifts_pair(self) -> np.ndarray:
        shift_probe = np.fft.fftshift(self.probe)
        probe_center_row = shift_probe[int(self.probe_shape[0]//2)]    
        probe_dia = np.count_nonzero(probe_center_row)

        start_shift = -(self.reconstruct_size-probe_dia)//2
        end_shift = (self.reconstruct_size-probe_dia)//2
        side_n_img = int(np.sqrt(self.n_img))
        shifts = np.linspace(start_shift, end_shift, side_n_img).astype(int)
        shifts_h, shifts_v = np.meshgrid(shifts, shifts)
        shifts_pair = np.concatenate([shifts_v.reshape(self.n_img,1), shifts_h.reshape(self.n_img,1)], axis=1)
        return shifts_pair

    def get_forward_model(self) -> BaseLinOp:
        probe = cp.array(self.probe)
        op_ifft2 = LinOpIFFT2() 
        op_fftshift = LinOpFFTSHIFT()
        op_ifftshift = LinOpIFFTSHIFT()
        op_fcrop = LinOpCrop2(self.reconstruct_size, self.probe_shape[0])
        op_probe = LinOpMul(probe)
        linop = StackLinOp([
            op_ifft2 @ op_probe @ op_ifftshift @ op_fcrop @ op_fftshift @ LinOpRoll2(self.shifts_pair[i_probe,0],self.shifts_pair[i_probe,1])
            for i_probe in range(self.n_img)
        ])
        return linop

    def get_probe_overlap_map(self) -> np.ndarray:
        pad_size    = self.reconstruct_size - self.probe_shape[0]
        shift_probe = np.fft.fftshift(self.probe)
        shift_probe = np.pad(shift_probe ,(int(np.floor(pad_size/2)), int(np.ceil(pad_size/2))), mode='constant')

        overlap_img = np.zeros(shape= (self.reconstruct_size, self.reconstruct_size))
        for i_probe in range(self.n_img):
            roll_linop  = LinOpRoll2(self.shifts_pair[i_probe,0],self.shifts_pair[i_probe,1])
            overlap_img = overlap_img + roll_linop.apply(shift_probe)
        return overlap_img
    
    def get_overlap_rate(self) -> float:
        """self-defined shifts_pair might cause error"""
        shift_probe = np.fft.fftshift(self.probe)
        probe_center_row = shift_probe[int(self.probe_shape[0]//2)]    
        probe_dia = np.count_nonzero(probe_center_row)
        probe_radius = probe_dia//2
        step_size = np.abs(self.shifts_pair[0][1] - self.shifts_pair[1][1])
        if step_size > (probe_radius*2):
            return 0
        else:
            circ_sector     = 2*(np.arccos(step_size/2/probe_radius)/(2*np.pi)) * np.pi*probe_radius**2
            tria_area       = step_size/2 * np.sqrt(probe_radius**2 - (step_size/2)**2)
            overlap_rate    = 2*(circ_sector - tria_area)/np.pi/(probe_radius**2)
            return overlap_rate