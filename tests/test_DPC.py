## Temporarily adding path
import sys
from pathlib import Path
sys.path.append(str(Path().absolute()))

## === Test Start ===
import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.patches as patches
from scipy import interpolate

from pyphaseretrieve.linop  import *
from pyphaseretrieve        import algos
from pyphaseretrieve        import phaseretrieval
from pyphaseretrieve        import loss

DAT_FILE_PATH = '/home/kshen/Ptychography_Project/phase-retrieval-library/phase-retrieval-library/dataset/DPC_dataset/' ## Local PATH


## ================================================== Dataset  ====================================================
## ================================================================================================================
class ptycho2d_Laura_dataSet(object):
    def __init__(self, camera_size:int) -> None:
        ## Experiment Setup Parameters Setting (distance and length unit: um)
        # Optical system
        self.wave_lambda        = 0.514
        self.NA                 = 0.2
        # Camera system
        self.camera_size        = camera_size
        self.mag                = 8.1485
        self.camera_pixel_Gsize = 6.5
        self.CAMERA_H_RES       = 2560
        self.CAMERA_V_RES       = 2160
        # LED system
        self.led_pitch          = 4000
        self.led_d_z            = 64000
        self.led_dia_number     = 19
        ## generate experimental setup
        self.experimentSetup()

    def experimentSetup(self):
        self.fourier_res           = self.get_fourierResolution()
        self.pupil_mask            = self.get_pupil_mask()

        self.NA_illu               = self.get_illumnationNA()

        self.total_shifts_h_map ,self.total_shifts_v_map , self.total_shifts_pair = self.get_shiftsMap_shiftsPairs()
    
    # ----------------------- methods -----------------------
    # -------------------------------------------------------
    def get_fourierResolution(self) -> tuple([float, float]):
        img_pixel_size         = self.camera_pixel_Gsize/self.mag
        FoV                    = img_pixel_size*self.camera_size
        fourier_resolution     = 1/FoV

        return fourier_resolution

    def get_pupil_mask(self):
        pupil_radius             = self.NA/self.wave_lambda/self.fourier_res

        camera_size_idx          = np.linspace(-math.floor(self.camera_size/2),math.ceil(self.camera_size/2)-1,self.camera_size) 
        temp_mask_h, temp_mask_v = np.meshgrid(camera_size_idx,camera_size_idx)

        pupil_mask, _ = cart2pol(temp_mask_h,temp_mask_v)
        pupil_mask[pupil_mask <= pupil_radius] = 1
        pupil_mask[pupil_mask >= pupil_radius] = 0

        return np.fft.fftshift(pupil_mask)

    def get_illumnationNA(self) -> float:
        # led_r_number    = math.floor(self.led_dia_number/2)
        # led_r           = led_r_number * self.led_pitch
        led_r           = 3 * self.led_pitch
        illuminationNA  = led_r/math.sqrt(self.led_d_z**2+led_r**2)

        return illuminationNA
        
    def get_reconstruction_size(self) -> int:
        synthetic_NA            = self.NA_illu + self.NA
        reconstruct_dia_number  = math.ceil(2*synthetic_NA/self.wave_lambda/self.fourier_res)

        return reconstruct_dia_number

    def get_shiftsMap_shiftsPairs(self, return_sinThetaMap_ledMask:bool = False):
        led_ra_size         = math.floor(self.led_dia_number/2)
        led_h_idx_array     = np.linspace(-led_ra_size,led_ra_size,self.led_dia_number)
        led_v_idx_array     = np.linspace(-led_ra_size,led_ra_size,self.led_dia_number)
        
        led_h_idx_map, led_v_idx_map = np.meshgrid(led_h_idx_array,led_v_idx_array)
        led_r_mask                   = np.sqrt(led_h_idx_map**2 + led_v_idx_map**2)
        led_r_mask[led_r_mask < self.led_dia_number/2] = 1
        led_r_mask[led_r_mask > self.led_dia_number/2] = 0

        n_used_led              = int(np.sum(led_r_mask))

        led_h_d_map             = led_h_idx_map * self.led_pitch
        led_v_d_map             = led_v_idx_map * self.led_pitch
        led_to_sample_dist_map  = np.sqrt( (led_h_d_map)**2 + (led_v_d_map)**2 + self.led_d_z**2)

        sinTheta_h_map      = led_h_d_map/led_to_sample_dist_map
        sinTheta_v_map      = led_v_d_map/led_to_sample_dist_map

        shifts_h_map = np.round(sinTheta_h_map * led_r_mask / self.wave_lambda / self.fourier_res)
        shifts_v_map = np.round(sinTheta_v_map * led_r_mask / self.wave_lambda / self.fourier_res)

        shifts_h    = shifts_h_map[led_r_mask == 1]
        shifts_v    = shifts_v_map[led_r_mask == 1]
        shifts_pair = np.concatenate([shifts_v.reshape(n_used_led,1),shifts_h.reshape(n_used_led,1)],axis=1)

        if return_sinThetaMap_ledMask:
            return sinTheta_h_map, sinTheta_v_map, led_r_mask
        else:
            return shifts_h_map, shifts_v_map, shifts_pair

    def get_bright_field_LED_map(self):
        sinTheta_h_map, sinTheta_v_map, led_r_mask = self.get_shiftsMap_shiftsPairs(return_sinThetaMap_ledMask = True)
        
        led_NA_map      = np.sqrt(sinTheta_h_map**2 + sinTheta_v_map**2)

        led_index_map   = np.ones((self.led_dia_number,self.led_dia_number))
        led_index_map   = led_index_map * led_r_mask
        led_index_map   = led_index_map.reshape(self.led_dia_number**2,)
        led_idx = 1
        for idx in range(self.led_dia_number**2):
            if led_index_map[idx] != 0:
                led_index_map[idx] = led_idx
                led_idx += 1
        led_index_map   = led_index_map.reshape(self.led_dia_number,self.led_dia_number)

        bright_field_led_map = ((led_NA_map <= 0.99 * self.NA) * led_index_map).astype(int)

        return bright_field_led_map

    def select_image(self, img_index_array, centre:list = [0, 0], load_img_or_not:bool = True, remove_background:bool = True, show_background:bool = False):
        shifts_pair     = self.total_shifts_pair[(img_index_array-1),0:2]
        
        file_path       = DAT_FILE_PATH

        h_start     = int(self.CAMERA_H_RES//2 + centre[0] - self.camera_size//2)
        v_start     = int(self.CAMERA_V_RES//2 - centre[1] - self.camera_size//2)
        crop_size   = self.camera_size
        if load_img_or_not:
            print('image loading...')
            img_list = []
            for i in img_index_array:
                file_name = str('ILED_{0:04}.tif'.format(i))
                img       = plt.imread(file_path + file_name)

                crop_img = img[v_start:v_start+crop_size, h_start:h_start+crop_size]
                img_list.append(crop_img)
            print('finish loading...')  

            if remove_background:
                img_background = img_list[0]
                for _, img in enumerate(img_list):
                    img_background = np.minimum(img_background, img)
                    
                if show_background:
                    plt.figure()
                    plt.imshow(img_background, cmap=cm.Greys_r)
                    plt.colorbar()
                    plt.title('Background image')
            else:
                img_background = 0

            y = None
            for _, _crop_img in enumerate(img_list):
                _crop_img_deback = _crop_img - img_background 
                if y is None:
                    y = _crop_img_deback
                else:
                    y = np.concatenate([y,_crop_img_deback], axis=0)
            
            return y, img_list, shifts_pair

        else:
            return None, None, shifts_pair
        
    # ----------------------- Render f ----------------------
    # -------------------------------------------------------
    def crop_rendering(self, centre:list = [0,0]) -> None:
        v_center        = self.CAMERA_V_RES/2 - centre[1]
        h_center        = self.CAMERA_H_RES/2 + centre[0]
        crop_half_size  = int(self.camera_size/2)

        file_path       = DAT_FILE_PATH
        file_name       = str('ILED_0147.tif')
        img             = plt.imread(file_path + file_name)

        _, ax = plt.subplots()
        ax.imshow(img, cmap=cm.Greys_r)
        ax.set_title('Center Bright field Image with cropping area')
        rect = patches.Rectangle((h_center-crop_half_size, v_center-crop_half_size), self.camera_size, self.camera_size, linewidth=2, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
        plt.title('Cropping')

    def pupil_rendering(self) -> None:
        plt.figure()
        plt.imshow(self.pupil_mask, cmap=cm.Greys_r)
        plt.colorbar()
        plt.title('Pupil mask')

    def total_shifts_rendering(self) -> None:
        plt.figure()
        plt.imshow(self.total_shifts_h_map, cmap=cm.Greys_r)
        plt.colorbar()
        plt.title('Total Horizental shifts map')

        plt.figure()
        plt.imshow(self.total_shifts_v_map, cmap=cm.Greys_r)
        plt.colorbar()
        plt.title('Total Vertical shifts map')
## ================================================ END Dataset ===================================================
## ================================================================================================================



## ================================================== Test Class ==================================================
## ================================================================================================================
class DPC_test(object):
    def __init__(self) -> None:
        pass

    def DPC_img(self, camera_size, n_iter= 1, GD_n_iter= 5,lr= 1):
        print('REAL dataset test \n----------------------')
        ## 1. use experimental setup from Laura dataset
        self.ptycho_data    = ptycho2d_Laura_dataSet(camera_size)
        bright_LED_map      = self.ptycho_data.get_bright_field_LED_map()
        bright_LED_array    = bright_LED_map[bright_LED_map>0]

        ## 2. ground truth x generating
        reconstruction_res          = camera_size
        # reconstruction_res          = self.ptycho_data.get_reconstruction_size()
        print(f'recontruction size: {reconstruction_res}')

        ## 4. ptycho2d model create
        img_idx_array               = bright_LED_array
        probe                       = self.ptycho_data.get_pupil_mask()
        y, img_list, shifts_pair    = self.ptycho_data.select_image(img_idx_array, remove_background= False)
        self.ptycho_2d_model        = phaseretrieval.FourierPtychography2d(probe= probe, shifts_pair= shifts_pair, reconstruct_size= reconstruction_res)

        sum_bright_img = np.uint32(np.zeros_like(img_list[0]))
        for idx in range(len(img_list)):
            sum_bright_img += img_list[idx]
        mean_bright_img = np.sum(sum_bright_img)/len(img_list)
        # print(mean_bright_img)
        ## 5. PPR solver
        initial_est             = np.ones(shape=(reconstruction_res,reconstruction_res), dtype=np.complex128)
        # initial_est             = np.ones(shape=(reconstruction_res,reconstruction_res), dtype=np.complex128)  + 1j * np.random.randn(reconstruction_res, reconstruction_res)
        initial_est             = np.fft.fft2(initial_est, norm="ortho")

        ppr_method              = algos.PerturbativePhase(self.ptycho_2d_model)
        x_est                   = ppr_method.iterate_GD(y = y, initial_est = initial_est, n_iter = n_iter, GD_n_iter= GD_n_iter, lr = lr)
        x_est                   = np.fft.ifft2(x_est, norm="ortho")

        ## 7. result
        plt.figure()
        plt.imshow(sum_bright_img, cmap=cm.Greys_r)
        plt.colorbar()
        plt.title('Sum Bright Field Image')
        plt.savefig('_recon_img/DPC_ Sum Bright Field Image.png')

        plt.figure()
        plt.imshow(img_list[int(len(img_list)/2)], cmap=cm.Greys_r)
        plt.colorbar()
        plt.title('Center Bright Field Image')
        plt.savefig('_recon_img/DPC_Center Bright Field Image.png')

        plt.figure()
        plt.imshow(np.abs(x_est)**2, cmap=cm.Greys_r)
        plt.colorbar()
        plt.title('Intensity: Reconstructed image')
        plt.savefig('_recon_img/DPC_Intensity: Reconstructed image.png')

        plt.figure()
        plt.imshow(np.angle(x_est), cmap=cm.Greys_r)
        plt.colorbar()
        plt.title('Phase: Reconstruction image')
        plt.savefig('_recon_img/DPC_Phase: Reconstruction image.png')
        return x_est

## ================================================== END Test Class ==================================================
## ====================================================================================================================


## ============ functions ============
def interpolate_img(img, high_res:int):
    img_size         = img.shape[0]
    
    x_idx_array      = np.arange(-math.floor(img_size/2),math.ceil(img_size/2),1)
    high_x_idx_array = np.arange(-math.floor(img_size/2),math.ceil(img_size/2),img_size/high_res)

    inter_img_f      = interpolate.interp2d(x_idx_array, x_idx_array, img, kind='linear')
    inter_img        = inter_img_f(high_x_idx_array, high_x_idx_array)

    return inter_img

def cart2pol(x, y):
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return(rho, phi)
## ============ functions ============



if __name__ == '__main__':

    DPC_obj = DPC_test()
    DPC_obj.DPC_img(camera_size= 256, n_iter= 1, GD_n_iter= 3,lr= 1e-2)
    