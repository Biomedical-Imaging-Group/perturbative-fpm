## Temporarily adding path
import sys
from pathlib import Path
sys.path.append(str(Path().absolute()))

## === Test Start ===
import os
import numpy as np
import cupy as cp
import math
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.patches as patches
from matplotlib_scalebar.scalebar import ScaleBar
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
import matplotlib.font_manager as fm
from scipy import interpolate

from pyphaseretrieve.linop  import *
from pyphaseretrieve        import algos
from pyphaseretrieve        import phaseretrieval
from pyphaseretrieve        import loss

DAT_FILE_PATH = '/home/kshen/Ptychography_Project/phase-retrieval-library/phase-retrieval-library/dataset/1LED/tif/' ## Local PATH


## ================================================== Dataset  ====================================================
## ================================================================================================================
class DogCardiac_dataset(object):
    def __init__(self, camera_size:int) -> None:
        ## Experiment Setup Parameters Setting (distance and length unit: um)
        # Optical system
        self.wave_lambda        = 0.6292
        self.NA                 = 0.1
        # Camera system
        self.camera_size        = camera_size
        self.mag                = 8.1485
        self.camera_pixel_Gsize = 6.5
        self.CAMERA_H_RES       = 2560
        self.CAMERA_V_RES       = 2160
        # LED system
        self.led_pitch          = 4000
        self.led_d_z            = 67500
        self.led_dia_number     = 19
        ## generate experimental setup
        self.experimentSetup()
    
    # Initialize experimental parameters
    def experimentSetup(self):
        self.fourier_res           = self.get_fourierResolution()
        self.pupil_mask            = self.get_pupil_mask()
        self.total_shifts_h_map ,self.total_shifts_v_map , self.total_shifts_pair = self.get_shiftsMap_shiftsPairs()

    # ----------------------- Default setup -----------------------
    def get_fourierResolution(self) -> float:
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
    
    # ----------------------- NA and reconstruction -----------------------
    def get_reconstruction_size(self, bright_field_NA:bool = False) -> int:
        if bright_field_NA:
            NA_illu = self.get_bright_illumnationNA()
        else:
            NA_illu = self.get_total_illumnationNA()
        synthetic_NA         = NA_illu + self.NA
        reconstruction_size  = math.ceil(2*synthetic_NA/self.wave_lambda/self.fourier_res)
        return reconstruction_size 

    def get_total_illumnationNA(self) -> float:
        led_r_number    = math.floor(self.led_dia_number/2)
        led_r           = led_r_number * self.led_pitch
        illuminationNA  = led_r/math.sqrt(self.led_d_z**2+led_r**2)
        return illuminationNA    

    def get_bright_illumnationNA(self) -> float:
        led_r_map,_                    = self.get_led_r_angle_map()
        _,birght_field_led_mask,_      = self.get_bright_field_LED_map()
        bright_field_led_radius_map    = led_r_map*birght_field_led_mask

        led_r           = int(np.max(bright_field_led_radius_map)) * self.led_pitch
        illuminationNA  = led_r/math.sqrt(self.led_d_z**2+led_r**2)
        return illuminationNA

    # ----------------------- LED map and mask -----------------------  
    def get_led_bool_mask(self) -> np.ndarray:
        led_ra_size         = math.floor(self.led_dia_number/2)
        led_h_idx_array     = np.linspace(-led_ra_size,led_ra_size,self.led_dia_number)
        led_v_idx_array     = np.linspace(-led_ra_size,led_ra_size,self.led_dia_number)
        
        led_h_idx_map, led_v_idx_map = np.meshgrid(led_h_idx_array,led_v_idx_array)
        led_r_map, _                 = cart2pol(led_h_idx_map,led_v_idx_map)
        led_bool_mask                     = led_r_map < self.led_dia_number/2
        return led_bool_mask

    def get_led_index_map(self) -> np.ndarray:
        led_bool_mask   = self.get_led_bool_mask()
        led_index_map   = np.ones((self.led_dia_number,self.led_dia_number))
        led_index_map   = led_index_map * led_bool_mask
        led_index_map   = led_index_map.reshape(self.led_dia_number**2,)
        led_idx = 1
        for idx in range(self.led_dia_number**2):
            if led_index_map[idx] != 0:
                led_index_map[idx] = led_idx
                led_idx += 1
        led_index_map   = led_index_map.reshape(self.led_dia_number,self.led_dia_number)
        return led_index_map

    def get_led_r_angle_map(self) -> np.ndarray:
        led_ra_size         = math.floor(self.led_dia_number/2)
        led_h_idx_array     = np.linspace(-led_ra_size,led_ra_size,self.led_dia_number)
        led_v_idx_array     = np.linspace(-led_ra_size,led_ra_size,self.led_dia_number)
        
        led_h_idx_map, led_v_idx_map = np.meshgrid(led_h_idx_array,led_v_idx_array)
        led_r_map, led_angle_map     = cart2pol(led_h_idx_map,led_v_idx_map)
        return led_r_map, -led_angle_map
    
    def get_led_360_coordinate_angle_map(self) -> np.ndarray:
        _, led_angle_map  = self.get_led_r_angle_map()
        led_360_anlge_map = np.copy(led_angle_map)
        led_360_anlge_map = np.flip(np.abs((led_360_anlge_map - np.abs(led_360_anlge_map))/2), axis=1)
        led_360_anlge_map[9,:] = 0
        led_angle_map[led_angle_map<-1] = 180
        led_angle_map     = led_angle_map + led_360_anlge_map
        return led_angle_map
        
    # ----------------------- Total shifts and shifts pairs ----------------------- 
    def get_shiftsMap_shiftsPairs(self, return_sinThetaMap_ledMap:bool = False):
        led_r_map, _    = self.get_led_r_angle_map()
        led_r_map[led_r_map < self.led_dia_number/2] = 1
        led_r_map[led_r_map > self.led_dia_number/2] = 0

        led_ra_size         = math.floor(self.led_dia_number/2)
        led_h_idx_array     = np.linspace(-led_ra_size,led_ra_size,self.led_dia_number)
        led_v_idx_array     = np.linspace(-led_ra_size,led_ra_size,self.led_dia_number)
        led_h_idx_map, led_v_idx_map = np.meshgrid(led_h_idx_array,led_v_idx_array)
        led_h_d_map             = led_h_idx_map * self.led_pitch
        led_v_d_map             = led_v_idx_map * self.led_pitch
        led_to_sample_dist_map  = np.sqrt( (led_h_d_map)**2 + (led_v_d_map)**2 + self.led_d_z**2)

        sinTheta_h_map      = led_h_d_map/led_to_sample_dist_map
        sinTheta_v_map      = led_v_d_map/led_to_sample_dist_map
    
        if return_sinThetaMap_ledMap:
            return sinTheta_h_map, sinTheta_v_map, led_r_map
        else:
            shifts_h_map = np.round(sinTheta_h_map * led_r_map / self.wave_lambda / self.fourier_res)
            shifts_v_map = np.round(sinTheta_v_map * led_r_map / self.wave_lambda / self.fourier_res)

            n_used_led  = int(np.sum(led_r_map))
            shifts_h    = shifts_h_map[led_r_map == 1]
            shifts_v    = shifts_v_map[led_r_map == 1]
            shifts_pair = np.concatenate([shifts_v.reshape(n_used_led,1),shifts_h.reshape(n_used_led,1)],axis=1)

            return shifts_h_map, shifts_v_map, shifts_pair
    # ----------------------- LEDs selection methods ----------------------- 
    def get_bright_field_LED_map(self):
        sinTheta_h_map, sinTheta_v_map, _ = self.get_shiftsMap_shiftsPairs(return_sinThetaMap_ledMap = True)
        led_NA_map                        = np.sqrt(sinTheta_h_map**2 + sinTheta_v_map**2)

        led_index_map                     = self.get_led_index_map()

        birght_field_led_bool_mask = (led_NA_map <= self.NA)
        birght_field_led_mask      = birght_field_led_bool_mask.astype(int)
        bright_field_led_index_map  = (birght_field_led_mask * led_index_map).astype(int)
        return bright_field_led_index_map, birght_field_led_mask, birght_field_led_bool_mask
    
    def select_image_by_imgIndex(self, img_index_array, centre:list = [0, 0], remove_background:bool = False, show_background:bool = False):
        print('image loading...')
        h_start     = int(self.CAMERA_H_RES//2 + centre[0] - self.camera_size//2)
        v_start     = int(self.CAMERA_V_RES//2 - centre[1] - self.camera_size//2)
        crop_size   = self.camera_size
        img_list = []
        for i in img_index_array:
            file_name = str('ILED_{0:04}.tif'.format(i))
            img       = plt.imread(DAT_FILE_PATH + file_name)

            crop_img = img[v_start:v_start+crop_size, h_start:h_start+crop_size]
            img_list.append(crop_img)

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
            _crop_img_demean = _crop_img - img_background 
            if y is None:
                y = _crop_img_demean
            else:
                y = np.concatenate([y,_crop_img_demean], axis=0)
        print('finish loading...')  
        
        shifts_pair = self.total_shifts_pair[(img_index_array-1),0:2]       
        return y, img_list, shifts_pair
        
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
class FPM_soler(object):
    def __init__(self) -> None:
        print('FPM GD Start\n======================')
        pass

    def generate_rand2d_x(self, in_dim):
        x = (cp.random.randn(in_dim, in_dim) + 1j * cp.random.randn(in_dim, in_dim))
        return x

    def simulation_test(self, camera_size:int, img_idx_array, n_iter:int, lr, spec_method:bool = False):
        print('Model test \n----------------------')
        self.dataset                = DogCardiac_dataset(camera_size= camera_size)

        reconstruction_res          = self.dataset.get_reconstruction_size()
        reconstruction_shape        = (reconstruction_res, reconstruction_res)
        print(f'recontruction shape: {reconstruction_shape}')

        x                       = self.generate_rand2d_x(reconstruction_res)
        x_ft                    = np.fft.fft2(x, norm="ortho")

        probe                   = cp.array(self.dataset.get_pupil_mask())
        shifts_pair             = self.dataset.total_shifts_pair[(img_idx_array-1),0:2]
        self.phase_model        = phaseretrieval.FourierPtychography2d(probe = probe, shifts_pair= shifts_pair, reconstruct_shape= reconstruction_shape)
        y                       = np.abs(self.phase_model.apply(x_ft))**2

        loss_function           = loss.loss_amplitude_based(epsilon=1e-1)
        GD_method               = algos.GradientDescent(self.phase_model, loss_func= loss_function, line_search= True, acceleration=None)
        Spec_method             = algos.SpectralMethod(self.phase_model)

        if spec_method:
            initial_est             = cp.random.randn(reconstruction_res,reconstruction_res)
            initial_est             = Spec_method.iterate(y= y, initial_est= initial_est)
        else:
            initial_est             = cp.ones(shape=(reconstruction_res,reconstruction_res), dtype=np.complex128) 
            initial_est             = np.fft.fft2(initial_est, norm="ortho")

        x_est                   = GD_method.iterate(y = y, initial_est = initial_est, n_iter = n_iter, lr = lr)
        x_est                   = np.fft.ifft2(x_est, norm="ortho")

        _x      = np.ravel(x)
        _x_est  = np.ravel(x_est)
        print("Result correlation:")
        print(np.abs( (_x_est.T.conj() @ _x) /  (np.linalg.norm(_x_est)*np.linalg.norm(_x)) ))
    
    def FPM(self, camera_size:int, n_iter:int, lr, centre:list= [0,0], img_idx_array:np.ndarray= None, amp_based_or_not:bool= True, spec_method:bool = False
            , for_loop_or_not:bool = False):
        print('FPM test \n----------------------')
        self.dataset            = DogCardiac_dataset(camera_size)

        reconstruction_res          = self.dataset.get_reconstruction_size()
        reconstruction_shape        = (reconstruction_res, reconstruction_res)
        print(f'recontruction shape: {reconstruction_shape}')

        if img_idx_array is not None:
            img_index_array = img_idx_array
        else:
            img_index_array = np.linspace(1,293,293).astype(int)

        probe                       = cp.array(self.dataset.get_pupil_mask())
        y, img_list, shifts_pair    = self.dataset.select_image_by_imgIndex(img_index_array= img_index_array, centre= centre)
        y                           = cp.array(y)
        self.phase_model        = phaseretrieval.FourierPtychography2d(probe= probe, shifts_pair= shifts_pair, reconstruct_shape= reconstruction_shape)

        if amp_based_or_not:
            epsilon                     = 1e-1
            loss_function               = loss.loss_amplitude_based(epsilon= epsilon)
            GD_method                   = algos.GradientDescent(self.phase_model, loss_func= loss_function, line_search= None)
        else:
            GD_method                   = algos.GradientDescent(self.phase_model, loss_func= None, line_search= True)

        if spec_method:
            Spec_method             = algos.SpectralMethod(self.phase_model)
            initial_est             = cp.random.randn(reconstruction_res,reconstruction_res)
            initial_est             = Spec_method.iterate(y= y, initial_est= initial_est)
        else:
            initial_est             = cp.ones(shape= reconstruction_shape, dtype=np.complex128) 
            initial_est             = np.fft.fft2(initial_est, norm="ortho")
        
        x_est                       = GD_method.iterate(y=y, initial_est=initial_est, n_iter = n_iter, lr = lr) 
        x_est                       = np.fft.ifft2(x_est, norm="ortho")

        if for_loop_or_not == False:
            plt.figure()
            plt.imshow(np.abs(x_est.get())**2, cmap=cm.Greys_r)
            plt.colorbar()
            plt.title('Intensity: Reconstructed image')
            plt.savefig('_recon_img/DogCardiac_FPM_Intensity image.png')

            plt.figure()
            plt.imshow(np.angle(x_est.get()), cmap=cm.Greys_r)
            plt.colorbar()
            plt.title('Phase: Reconstruction image')
            plt.savefig('_recon_img/DogCardiac_FPM_Phase image.png')
        else:
            file_path   = str(f'_FPM/amp_based/n_iter={n_iter}')
            file_header = str('/DogCardiac_FPM_')
            file_iter   = str(f'n_iter={n_iter}, lr= {lr}, epsilon={epsilon}.png')
            plt.figure()
            plt.imshow(np.abs(x_est.get())**2, cmap=cm.Greys_r)
            plt.colorbar()
            plt.title(f'Intensity: '+ file_iter)
            plt.savefig( file_path + file_header + 'Intensity: ' + file_iter)
            plt.close()

            plt.figure()
            plt.imshow(np.angle(x_est.get()), cmap=cm.Greys_r)
            plt.colorbar()
            plt.title(f'Phase: '+ file_iter)
            plt.savefig( file_path + file_header + 'Phase: ' + file_iter)
            plt.close()

            phase_img = np.angle(x_est)
            phase_img_99 = np.percentile(np.abs(phase_img),99.99)
            phase_img[phase_img > phase_img_99]  = 0
            phase_img[phase_img < -phase_img_99] = 0
            plt.figure()
            plt.imshow(phase_img.get(), cmap=cm.Greys_r)
            plt.colorbar()
            plt.title(f'Phase Image, Filter out 99.99%: ' + file_iter)
            plt.savefig( file_path + file_header + 'Phase99: ' + file_iter)
            plt.close()

            histogram, bin_edges = np.histogram(np.angle(x_est.get()), bins= (x_est.shape[0]*x_est.shape[1]))
            plt.figure()
            plt.xlabel("grayscale value")
            plt.ylabel("pixel count")
            plt.plot(bin_edges[0:-1], histogram)
            _, max_ylim = plt.ylim()
            plt.text(phase_img_99.get()*1.1, max_ylim*0.9, '99.99%= {:.2f}'.format(phase_img_99.get()))
            plt.axvline(phase_img_99.get(), color='k', linestyle='dashed', linewidth=1)
            plt.title(f'Phase Grayscale Histogram: ' + file_iter)
            plt.savefig( file_path + file_header + 'Phase Histogram: ' + file_iter)
            plt.close()

        return x_est
    
    def FPM_PPR(self, camera_size:int, n_iter:int, linear_n_iter, lr, centre:list= [0,0], img_idx_array:np.ndarray= None, for_loop_or_not:bool = False):
        print('FPM test \n----------------------')
        self.dataset                = DogCardiac_dataset(camera_size)

        reconstruction_res          = self.dataset.get_reconstruction_size()
        reconstruction_shape        = (reconstruction_res, reconstruction_res)
        print(f'recontruction shape: {reconstruction_shape}')

        if img_idx_array is not None:
            img_index_array = img_idx_array
        else:
            img_index_array = np.linspace(1,293,293).astype(int)

        probe                       = cp.array(self.dataset.get_pupil_mask())
        y, img_list, shifts_pair    = self.dataset.select_image_by_imgIndex(img_index_array= img_index_array, centre= centre)
        y                           = cp.array(y)
        self.phase_model            = phaseretrieval.FourierPtychography2d(probe= probe, shifts_pair= shifts_pair, reconstruct_shape= reconstruction_shape)

        initial_est                 = cp.ones(shape= reconstruction_shape, dtype=np.complex128) 
        initial_est                 = np.fft.fft2(initial_est, norm="ortho")

        ppr_method                  = algos.PerturbativePhase(self.phase_model)
        if lr is not None:
            x_est                   = ppr_method.iterate_GradientDescent(y= y, initial_est= initial_est, n_iter= n_iter, linear_n_iter= linear_n_iter, lr=lr)
        else:
            x_est                   = ppr_method.iterate_ConjugateGradientDescent(y= y, initial_est= initial_est, n_iter= n_iter, linear_n_iter= linear_n_iter)
        
        x_est                       = np.fft.ifft2(x_est, norm="ortho")

        if for_loop_or_not == False:
            plt.figure()
            plt.imshow(np.abs(x_est.get())**2, cmap=cm.Greys_r)
            plt.colorbar()
            plt.title('Intensity: Reconstructed image')
            plt.savefig('_recon_img/DogCardiac_FPM_Intensity image.png')

            plt.figure()
            plt.imshow(np.angle(x_est.get()), cmap=cm.Greys_r)
            plt.colorbar()
            plt.title('Phase: Reconstruction image')
            plt.savefig('_recon_img/DogCardiac_FPM_Phase image.png')
        else:
            file_path   = str(f'_FPM_PPR/CGD/n_iter={n_iter}')
            file_header = str('/DogCardiac_FPM_')
            file_iter   = str(f'linear_n_iter={linear_n_iter}.png')
            plt.figure()
            plt.imshow(np.abs(x_est.get())**2, cmap=cm.Greys_r)
            plt.colorbar()
            plt.title(f'Intensity: '+ file_iter)
            plt.savefig( file_path + file_header + 'Intensity: ' + file_iter)
            plt.close()

            plt.figure()
            plt.imshow(np.angle(x_est.get()), cmap=cm.Greys_r)
            plt.colorbar()
            plt.title(f'Phase: '+ file_iter)
            plt.savefig( file_path + file_header + 'Phase: ' + file_iter)
            plt.close()

            phase_img = np.angle(x_est)
            phase_img_99 = np.percentile(np.abs(phase_img),99.99)
            phase_img[phase_img > phase_img_99]  = 0
            phase_img[phase_img < -phase_img_99] = 0
            plt.figure()
            plt.imshow(phase_img.get(), cmap=cm.Greys_r)
            plt.colorbar()
            plt.title(f'Phase Image, Filter out 99.99%: ' + file_iter)
            plt.savefig( file_path + file_header + 'Phase99: ' + file_iter)
            plt.close()

            histogram, bin_edges = np.histogram(np.angle(x_est.get()), bins= (x_est.shape[0]*x_est.shape[1]))
            plt.figure()
            plt.xlabel("grayscale value")
            plt.ylabel("pixel count")
            plt.plot(bin_edges[0:-1], histogram)
            _, max_ylim = plt.ylim()
            plt.text(phase_img_99.get()*1.1, max_ylim*0.9, '99.99%= {:.2f}'.format(phase_img_99.get()))
            plt.axvline(phase_img_99.get(), color='k', linestyle='dashed', linewidth=1)
            plt.title(f'Phase Grayscale Histogram: ' + file_iter)
            plt.savefig( file_path + file_header + 'Phase Histogram: ' + file_iter)
            plt.close()

        return x_est
    
    def Bright_FPM(self, camera_size:int, n_iter:int, lr, centre:list= [0,0], spec_method:bool = False):
        print('Bright FPM test \n----------------------')
        self.dataset                = DogCardiac_dataset(camera_size)

        reconstruction_res          = self.dataset.get_reconstruction_size(bright_field_NA= True)
        if reconstruction_res < camera_size:
            reconstruction_res = camera_size
        reconstruction_shape        = (reconstruction_res, reconstruction_res)
        print(f'recontruction shape: {reconstruction_shape}')

        bright_field_led_index_map, _, _ = self.dataset.get_bright_field_LED_map()
        bright_LED_index_array           = bright_field_led_index_map[bright_field_led_index_map>0]
        probe                       = cp.array(self.dataset.get_pupil_mask())
        y, img_list, shifts_pair    = self.dataset.select_image_by_imgIndex(img_index_array= bright_LED_index_array, centre= centre)
        y                           = cp.array(y)
        self.phase_model            = phaseretrieval.FourierPtychography2d(probe= probe, shifts_pair= shifts_pair, reconstruct_shape= reconstruction_shape)

        loss_function               = loss.loss_amplitude_based(epsilon=1e-4)
        GD_method                   = algos.GradientDescent(self.phase_model, loss_func= loss_function)
        Spec_method                 = algos.SpectralMethod(self.phase_model)

        if spec_method:
            initial_est             = cp.random.randn(reconstruction_res,reconstruction_res)
            initial_est             = Spec_method.iterate(y= y, initial_est= initial_est)
        else:
            initial_est             = cp.ones(shape=(reconstruction_res,reconstruction_res), dtype=np.complex128) 
            initial_est             = np.fft.fft2(initial_est, norm="ortho")
        
        x_est                       = GD_method.iterate(y=y, initial_est=initial_est, n_iter = n_iter, lr = lr) 
        x_est                       = np.fft.ifft2(x_est, norm="ortho")

        plt.figure()
        plt.imshow(np.abs(x_est.get())**2, cmap=cm.Greys_r)
        plt.colorbar()
        plt.title('Intensity: Reconstructed image')
        plt.savefig('_recon_img/DogCardiac_Bright_FPM_Intensity image.png')

        plt.figure()
        plt.imshow(np.angle(x_est.get()), cmap=cm.Greys_r)
        plt.colorbar()
        plt.title('Phase: Reconstruction image')
        plt.savefig('_recon_img/DogCardiac_Bright_FPM_Phase image.png')
        return x_est
    
    def phase_model_autoshifts_test(self, camera_size:int, n_img:int, n_iter:int, lr, spec_method:bool = False) -> None:
        print('Model test without shifts assignment \n----------------------')
        self.dataset        = DogCardiac_dataset(camera_size)

        reconstruction_res          = self.dataset.get_reconstruction_size()
        reconstruction_shape        = (reconstruction_res, reconstruction_res)
        print(f'recontruction shape: {reconstruction_shape}')

        x                       = self.generate_rand2d_x(reconstruction_res)
        x_ft                    = np.fft.fft2(x, norm="ortho")

        probe                   = cp.array(self.dataset.get_pupil_mask())
        self.phase_model        = phaseretrieval.FourierPtychography2d(probe= probe, reconstruct_shape= reconstruction_shape, n_img= n_img)
        print(f'overlap rate: {self.phase_model.get_overlap_rate()}')
        y                       = cp.abs(self.phase_model.apply(x_ft))**2

        loss_function           = loss.loss_amplitude_based(epsilon=1e-1)
        GD_method               = algos.GradientDescent(self.phase_model, loss_func= loss_function, line_search= None, acceleration=None)
        Spec_method             = algos.SpectralMethod(self.phase_model)

        if spec_method:
            initial_est             = cp.random.randn(reconstruction_res,reconstruction_res)
            initial_est             = Spec_method.iterate(y= y, initial_est= initial_est)
        else:
            initial_est             = cp.ones(shape=(reconstruction_res,reconstruction_res), dtype=np.complex128) 
            initial_est             = np.fft.fft2(initial_est, norm="ortho")

        x_est                   = GD_method.iterate(y= y, initial_est= initial_est, n_iter= n_iter, lr= lr)
        x_est                   = np.fft.ifft2(x_est, norm="ortho")

        _x      = np.ravel(x)
        _x_est  = np.ravel(x_est)
        print("Result correlation:")
        print(np.abs( (_x_est.T.conj() @ _x) /  (np.linalg.norm(_x_est)*np.linalg.norm(_x)) ))

        _x_init  = np.ravel(np.fft.ifft2(initial_est, norm="ortho"))
        print("Result without optimize correlation:")
        print(np.abs( (_x_init.T.conj() @ _x) /  (np.linalg.norm(_x_init)*np.linalg.norm(_x)) ))

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

def delete_file(dir_name):
    dir_name = dir_name
    for file_name in os.listdir(dir_name):
        os.remove(os.path.join(dir_name, file_name))

def cart2pol(x, y):
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return(rho, phi)
## ============ functions ============



if __name__ == '__main__':
    ## clean folder
    # delete_file('led_pattern')
    delete_file('_recon_img')   
    # ====================================================================================================
    # ====================================================================================================
    ## 1. FPM
    FPM_test = FPM_soler()

    # Test 1: model test
    img_idx_array = np.linspace(1,293,293).astype(int)
    # FPM_test.simulation_test(camera_size= 100, img_idx_array= img_idx_array, n_iter= 100, lr= 1e-2)

    # Test 2: real data
    centre = [-50,450]
    FPM_test.FPM(camera_size= 256, centre= centre, n_iter= 0,lr= 1, amp_based_or_not=False, spec_method= False)
    # for n_iter in [1,2,3,4,5,10,15,20,25]:
    #     delete_file(f'_FPM/amp_based/n_iter={n_iter}')
    #     for lr in np.geomspace(1, 1e-7, num=8):
    #         FPM_test.FPM(camera_size= 256, centre= centre, n_iter= n_iter,lr= lr, amp_based_or_not=True, spec_method= False, for_loop_or_not= True)
    # for n_iter in [2,4,6,8,10,15]:
    #     for linear_n_iter in [2,4,6,8,10]:
    #         FPM_test.FPM_PPR(camera_size= 256, centre= centre, n_iter= n_iter, linear_n_iter= linear_n_iter,lr= None, for_loop_or_not= True)
    # FPM_test.FPM_PPR(camera_size= 256, centre= centre, n_iter= 5, linear_n_iter= 9,lr= 1e-6)
    # FPM_test.Bright_FPM(camera_size= 256, centre= centre, n_iter= 15,lr= 1, spec_method= False)

    # Test 3: auto shift
    # ptycho2d_test.phase_model_autoshifts_test(camera_size= 64, n_img= 17**2, n_iter= 500, lr= 0.046)  # 64, 17**2, 200, 0.0457
