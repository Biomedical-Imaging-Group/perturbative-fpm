from abc import abstractmethod
import numpy as np
from pyphaseretrieve.linop import*

class PhaseRetrievalBase:
    def __init__(self, linop:BaseLinOp):
        self.linop = linop

    def apply(self, x):
        return self.linop.apply(x)
    
    def apply_ModularSquare(self, x):
        return np.abs(self.linop.apply(x))**2

    def applyT(self, x):
        return self.linop.applyT(x)

class Ptychography1d(PhaseRetrievalBase):
    def __init__(self, probe, shifts:np.ndarray=None, n_img:int=10):
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
        op_fft = LinOpFFT()
        op_probe = LinOpMul(self.probe)
        linop = StackLinOp([
            op_fft @ op_probe @ LinOpRoll(self.shifts[i_probe])
            for i_probe in range(self.n_img)
        ])
        return linop
    
    def get_probe_overlap_array(self) -> np.ndarray:
        overlap_img = np.zeros(shape= self.probe_shape)
        for i_probe in range(self.n_img):
            roll_linop  = LinOpRoll(-self.shifts[i_probe])
            overlap_img = overlap_img + roll_linop.apply(self.probe)
        return overlap_img

    def overlap_rate(self) -> float:
        probe_dia = np.count_nonzero(self.probe)
        step_size = np.abs(self.shifts[0]-self.shifts[1])
        overlap = 1 - step_size / probe_dia
        return overlap

class FourierPtychography2d(PhaseRetrievalBase):
    def __init__(self, probe, shifts_pair:np.ndarray= None, reconstruct_size:int= None, n_img:int= 25):
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
        op_ifft2 = LinOpIFFT2() 
        op_fftshift = LinOpFFTSHIFT()
        op_ifftshift = LinOpIFFTSHIFT()
        op_fcrop = LinOpCrop2(self.reconstruct_size, self.probe_shape[0])
        op_probe = LinOpMul(self.probe)
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
            roll_linop  = LinOpRoll2(-self.shifts_pair[i_probe,0],-self.shifts_pair[i_probe,1])
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

class XRay_Ptychography2d(PhaseRetrievalBase):
    def __init__(self, probe, shifts_pair:np.ndarray= None, reconstruct_shape:tuple= None, n_img:int= 25):
        """shifts_pair is defined as [v_shifts,h_shifts]"""
        self.probe = probe
        self.probe_shape = probe.shape

        if reconstruct_shape is not None:
            self.reconstruct_shape = reconstruct_shape
        else:
            self.reconstruct_shape = self.probe_shape

        assert shifts_pair.ndim == 2 , "shifts_map dimension should be (n,2)"
        self.n_img = shifts_pair.shape[0]
        self.shifts_pair = shifts_pair
            
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
        op_fft2 = LinOpFFT2()
        op_ifftshift = LinOpIFFTSHIFT()
        op_fcrop = LinOpCrop2_NonSquare(in_shape= self.reconstruct_shape, crop_shape= self.probe_shape)
        op_probe = LinOpMul(self.probe)
        linop = StackLinOp([
            op_ifftshift @ op_fft2 @ op_probe @ op_fcrop @ LinOpRoll2_PadZero(self.shifts_pair[i_probe,0],self.shifts_pair[i_probe,1])
            for i_probe in range(self.n_img)
        ])
        return linop

    def get_probe_overlap_map(self) -> np.ndarray:
        op_fcrop = LinOpCrop2_NonSquare(in_shape= self.reconstruct_shape, crop_shape= self.probe_shape)
        shift_probe = op_fcrop.applyT(self.probe)

        overlap_img = np.zeros_like(shift_probe)
        for i_probe in range(self.n_img):
            roll_linop  = LinOpRoll2_PadZero(-self.shifts_pair[i_probe,0],-self.shifts_pair[i_probe,1])
            overlap_img = overlap_img + roll_linop.apply(shift_probe)
        return overlap_img

class MultiplexedPhaseRetrieval(PhaseRetrievalBase):
    def __init__(self, probe, multiplex_led_mask:np.ndarray, shifts_pair:np.ndarray= None, reconstruct_size= None):
        """shifts_pair is defined as [v_shifts,h_shifts]"""
        self.probe = probe
        self.probe_shape = probe.shape
        self.multiplex_led_mask = multiplex_led_mask

        if reconstruct_size is not None:
            self.reconstruct_size = reconstruct_size
        else:
            self.reconstruct_size = self.probe_shape

        assert shifts_pair.ndim == 2 , "shifts_map dimension should be (n,2)"
        self.n_img = self.multiplex_led_mask.shape[0]
        self.shifts_pair = shifts_pair

        self.linop_array = self.get_linop_array()
        self.in_shape = (self.probe_shape[0]*self.n_img, self.probe_shape[1])
    
    def get_linop_array(self) -> BaseLinOp:
        op_ifft2 = LinOpIFFT2() 
        op_fftshift = LinOpFFTSHIFT()
        op_ifftshift = LinOpIFFTSHIFT()
        op_fcrop = LinOpCrop2(self.reconstruct_size, self.probe_shape[0])
        op_probe = LinOpMul(self.probe)

        total_linop_list = []
        for i_probe in range(self.shifts_pair.shape[0]):
            _linop = op_ifft2 @ op_probe @ op_ifftshift @ op_fcrop @ op_fftshift @ LinOpRoll2(self.shifts_pair[i_probe,0],self.shifts_pair[i_probe,1])
            total_linop_list.append(_linop)
        total_linop_array = np.array(total_linop_list)
        
        linop_array = total_linop_array*self.multiplex_led_mask
        return linop_array
    
    def apply(self, x):
        raise NameError('No apply method in MultiplexedPhaseRetrieval')

    def apply_ModularSquare(self, x):
        y_est = None
        for _, i_array in enumerate(self.linop_array):
            single_y = 0
            for _,i_linop in enumerate(i_array):
                single_y += np.abs(i_linop.apply(x))**2
            if y_est is None:
                y_est = np.copy(single_y)
            else:
                y_est = np.concatenate((y_est, single_y), axis=0)
        return y_est

    def get_perturbative_model(self, x_est):
        perturbative_model_list = []
        for _, i_array in enumerate(self.linop_array):
            _perturbative_model = None
            for _,i_linop in enumerate(i_array):
                _out_field = i_linop.apply(x_est)
                if _perturbative_model is None:
                    _perturbative_model = 2 * LinOpReal() @ LinOpMul(_out_field.conj()) @ i_linop
                else:
                    _perturbative_model += 2 * LinOpReal() @ LinOpMul(_out_field.conj()) @ i_linop
            perturbative_model_list.append(_perturbative_model)
        perturbative_model = StackLinOp(perturbative_model_list)
        return perturbative_model

    def get_probe_overlap_map(self) -> np.ndarray:
        pad_size    = self.reconstruct_size - self.probe_shape[0]
        shift_probe = np.fft.fftshift(self.probe)
        shift_probe = np.pad(shift_probe ,(int(np.floor(pad_size/2)), int(np.ceil(pad_size/2))), mode='constant')

        overlap_img = np.zeros(shape= (self.reconstruct_size, self.reconstruct_size))
        for i_probe in range(self.n_img):
            roll_linop  = LinOpRoll2(-self.shifts_pair[i_probe,0],-self.shifts_pair[i_probe,1])
            overlap_img = overlap_img + roll_linop.apply(shift_probe)
        return overlap_img