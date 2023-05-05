import numpy as np
from pyphaseretrieve.base_linop import BaseLinOp

## 1D classes
class LinOpMatrix(BaseLinOp):
    def __init__(self, matrix):
        self.H = matrix
        self.in_shape = (matrix.shape[1],)
        self.out_shape = (matrix.shape[0],)
    
    def apply(self, x):
        return self.H @ x

    def applyT(self, x):
        return self.H.T.conj() @ x

class LinOpMul(BaseLinOp):
    """coefs is for element-wise multiplication"""
    def __init__(self, coefs):
        self.coefs = coefs
        self.in_shape = coefs.shape
        self.out_shape = coefs.shape

    def apply(self, x):
        return self.coefs * x

    def applyT(self, x):
        return self.coefs.conj() * x

class LinOpFFT(BaseLinOp):
    def __init__(self):
        self.in_shape = (-1,)
        self.out_shape = (-1,)

    def apply(self, x):
        return np.fft.fft(x, norm="ortho")

    def applyT(self, x):
        return np.fft.ifft(x, norm="ortho")

class LinOpIFFT(BaseLinOp):
    def __init__(self):
        self.in_shape = (-1,)
        self.out_shape = (-1,)

    def apply(self, x):
        return np.fft.ifft(x, norm="ortho")

    def applyT(self, x):
        return np.fft.fft(x, norm="ortho")

class LinOpFFTSHIFT(BaseLinOp):
    def __init__(self):
        self.in_shape = (-1,)
        self.out_shape = (-1,)

    def apply(self, x):
        return np.fft.fftshift(x)

    def applyT(self, x):
        return np.fft.ifftshift(x)

class LinOpIFFTSHIFT(BaseLinOp):
    def __init__(self):
        self.in_shape = (-1,)
        self.out_shape = (-1,)

    def apply(self, x):
        return np.fft.ifftshift(x)

    def applyT(self, x):
        return np.fft.fftshift(x)

class LinOpId(BaseLinOp):
    def __init__(self):
        self.in_shape = (-1,)
        self.out_shape = (-1,)
        
    def apply(self, x):
        return x
    
    def applyT(self, x):
        return x
    
class LinOpFlip(BaseLinOp):
    def __init__(self):
        self.in_shape = (-1,)
        self.out_shape = (-1,)

    def apply(self, x):
        return np.flip(x)

    def applyT(self, x):
        return np.flip(x)

class LinOpRoll(BaseLinOp):
    def __init__(self, shifts):
        self.in_shape = (-1,)
        self.out_shape = (-1,)
        self.shifts = int(shifts)

    def apply(self, x):
        return np.roll(x, shift=self.shifts, axis=0)

    def applyT(self, x):
        return np.roll(x, shift=-self.shifts, axis=0)

class LinOpReal(BaseLinOp):
    def __init__(self):
        self.in_shape = (-1,)
        self.out_shape = (-1,)
        
    def apply(self, x):
        return np.real(x)

    def applyT(self, x):
        return x

class LinOpImag(BaseLinOp):
    def __init__(self):
        self.in_shape = (-1,)
        self.out_shape = (-1,)

    def apply(self, x):
        return np.imag(x)

    def applyT(self, x):
        return np.zeros_like(x, dtype= np.complex128)

class LinOp_RealPartExpand(BaseLinOp):
    def __init__(self, LinOp:BaseLinOp):
        self.LinOp = LinOp
        self.in_shape = (2*LinOp.in_shape[0],)
        self.out_shape = LinOp.out_shape
    
    def apply(self,x):
        return np.real(self.LinOp.apply(x[0:self.LinOp.in_shape[0]])) - np.imag(self.LinOp.apply(x[-self.LinOp.in_shape[0]:]) )

    def applyT(self,x):
        return np.concatenate((np.real(self.LinOp.applyT(x)), np.imag(self.LinOp.applyT(x))), axis=0)

## 2D classes
class LinOpMatrix2(BaseLinOp):
    def __init__(self, matrix):
        self.H = matrix
        self.in_shape = (-1,)
        self.out_shape = (-1,)
    
    def apply(self, x):
        return self.H @ x

    def applyT(self, x):
        return self.H.T.conj() @ x

class LinOpFFT2(BaseLinOp):
    def __init__(self):
        self.in_shape = (-1,)
        self.out_shape = (-1,)

    def apply(self, x):
        return np.fft.fft2(x, norm="ortho")

    def applyT(self, x):
        return np.fft.ifft2(x, norm="ortho")

class LinOpIFFT2(BaseLinOp):
    def __init__(self):
        self.in_shape = (-1,)
        self.out_shape = (-1,)

    def apply(self, x):
        return np.fft.ifft2(x, norm="ortho")

    def applyT(self, x):
        return np.fft.fft2(x, norm="ortho")

class LinOpId2(BaseLinOp):
    def __init__(self):
        self.in_shape = (-1,)
        self.out_shape = (-1,)
        
    def apply(self, x):
        return x
    
    def applyT(self, x):
        return x
    
class LinOpFlip2(BaseLinOp):
    def __init__(self):
        self.in_shape = (-1,)
        self.out_shape = (-1,)

    def apply(self, x):
        return np.flip(x)

    def applyT(self, x):
        return np.flip(x)

class LinOpRoll2(BaseLinOp):
    def __init__(self, v_shifts, h_shifts):
        self.in_shape = (-1,)
        self.out_shape = (-1,)
        self.v_shifts = int(v_shifts)
        self.h_shifts = int(h_shifts)

    def apply(self, x):
        return np.roll(x, shift=(self.v_shifts,self.h_shifts), axis=(0,1))

    def applyT(self, x):
        return np.roll(x, shift=(-self.v_shifts,-self.h_shifts), axis=(0,1))

class LinOpCrop2(BaseLinOp):
    def __init__(self, in_shape, crop_shape):
        """assume square size of input image"""
        self.in_shape = in_shape
        self.out_shape = crop_shape
        self.crop_shape = crop_shape

    def apply(self, x):
        v_size, h_size = x.shape
        v_start = int(v_size//2 - (self.crop_shape[0]//2))
        h_start = int(h_size//2 - (self.crop_shape[1]//2))
        return x[v_start:v_start+self.crop_shape[0], h_start:h_start+self.crop_shape[1]]

    def applyT(self, x):
        v_pad_size = self.in_shape[0] - self.crop_shape[0]        
        h_pad_size = self.in_shape[1] - self.crop_shape[1] 

        if v_pad_size != 0:
            if self.in_shape[0]%2 == 1:
                x = np.pad(x,((int(np.floor(v_pad_size/2)), int(np.ceil(v_pad_size/2))), (0, 0)),mode='constant')
            else:
                x = np.pad(x,((int(np.ceil(v_pad_size/2)), int(np.floor(v_pad_size/2))), (0, 0)),mode='constant')

        if h_pad_size != 0:
            if self.in_shape[1]%2 == 1:
                x = np.pad(x,((0, 0), (int(np.floor(h_pad_size/2)), int(np.ceil(h_pad_size/2)))),mode='constant')
            else:
                x = np.pad(x,((0, 0), (int(np.ceil(h_pad_size/2)), int(np.floor(h_pad_size/2)))),mode='constant')
        return x

class LinOpRoll2_PadZero(BaseLinOp):
    def __init__(self, v_shifts, h_shifts):
        self.in_shape = (-1,)
        self.out_shape = (-1,)
        self.v_shifts = int(v_shifts)
        self.h_shifts = int(h_shifts)

    def apply(self, x):
        x = np.roll(x, self.h_shifts, axis=1)
        if self.h_shifts < 0:
            x[:, self.h_shifts:] = 0
        elif self.h_shifts > 0:
            x[:, 0:self.h_shifts] = 0

        x = np.roll(x, self.v_shifts , axis=0)
        if self.v_shifts  < 0:
            x[self.v_shifts :, :] = 0
        elif self.v_shifts  > 0:
            x[0:self.v_shifts , :] = 0
        return x

    def applyT(self, x):
        x = np.roll(x, -self.h_shifts, axis=1)
        if -self.h_shifts < 0:
            x[:, -self.h_shifts:] = 0
        elif -self.h_shifts > 0:
            x[:, 0:-self.h_shifts] = 0

        x = np.roll(x, -self.v_shifts , axis=0)
        if -self.v_shifts  < 0:
            x[-self.v_shifts :, :] = 0
        elif -self.v_shifts  > 0:
            x[0:-self.v_shifts , :] = 0
        return x

## Dimensionless 
class StackLinOp(BaseLinOp):
    def __init__(self, LinOpList):
        self.LinOpList = LinOpList
        self.in_shape = max([linop.in_shape for linop in LinOpList])
        self.out_shape = tuple(np.sum(np.array([(linop.out_shape if linop.out_shape>(0,) else self.in_shape)
            for linop in LinOpList]),axis=0))

    def apply(self, x):
        return np.concatenate(tuple(linop.apply(x) for linop in self.LinOpList), axis=0)

    def applyT(self, x):       
        current_idx = 0
        for idx, linop in enumerate(self.LinOpList):
            if idx == 0:
                res = linop.applyT(x[current_idx:current_idx+linop.out_shape[0]])
            res += linop.applyT(x[current_idx:current_idx+linop.out_shape[0]])
            current_idx += (linop.out_shape[0] if linop.out_shape[0]>0 else self.in_shape[0])
        return res

## functions
def shift_2d_replace(data, dx, dy, constant=False):
    shifted_data = np.roll(data, dx, axis=1)
    if dx < 0:
        shifted_data[:, dx:] = constant
    elif dx > 0:
        shifted_data[:, 0:dx] = constant

    shifted_data = np.roll(shifted_data, dy, axis=0)
    if dy < 0:
        shifted_data[dy:, :] = constant
    elif dy > 0:
        shifted_data[0:dy, :] = constant
    return shifted_data