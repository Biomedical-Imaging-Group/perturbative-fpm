import numpy as np
from typing import List
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

class RealPartExpandOp(BaseLinOp):
    def __init__(self, LinOp:BaseLinOp):
        self.LinOp = LinOp
        self.in_size = (2*LinOp.in_shape[0],)
        self.out_size = LinOp.out_shape
    
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
    def __init__(self, in_size, crop_size):
        """assume square size of input image"""
        self.in_shape = (in_size,in_size)
        self.out_shape = (crop_size,crop_size)
        self.in_size  = in_size
        self.crop_size = crop_size

    def apply(self, x):
        v_size, h_size = x.shape
        h_start = int(h_size//2 - (self.crop_size//2))
        v_start = int(v_size//2 - (self.crop_size//2))
        return x[v_start:v_start+self.crop_size, h_start:h_start+self.crop_size]

    def applyT(self, x):
        pad_size = self.in_size - self.crop_size        
        if      pad_size == 0:
            return x
        else:            
            return np.pad(x ,(int(np.floor(pad_size/2)), int(np.ceil(pad_size/2))), mode='constant')

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
        res = np.zeros(self.in_shape).astype(np.complex128)
        for linop in self.LinOpList:
            res += linop.applyT(x[current_idx:current_idx+linop.out_shape[0]])
            current_idx += (linop.out_shape[0] if linop.out_shape[0]>0 else self.in_shape[0])
        return res