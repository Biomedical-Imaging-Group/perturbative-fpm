## Temporarily adding path
import sys
from pathlib import Path
sys.path.append(str(Path().absolute()))

## === Test Start ===
import numpy as np
import cupy as cp
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
        self.total_n_img        = 293
        ## generate experimental setup
        self.experimentSetup()

    def experimentSetup(self):
        self.fourier_res           = self.get_fourierResolution()
        self.pupil_mask            = self.get_pupil_mask()

        # self.NA_illu               = self.get_illumnationNA()
        self.NA_illu               = self.get_bright_illumnationNA()

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
        led_r_number    = math.floor(self.led_dia_number/2)
        led_r           = led_r_number * self.led_pitch
        illuminationNA  = led_r/math.sqrt(self.led_d_z**2+led_r**2)
        return illuminationNA
    
    def get_bright_illumnationNA(self) -> float:
        led_r_map,_ = self.get_led_r_angle_map()
        
        _,birght_field_led_mask      = self.get_bright_field_LED_map()
        bright_field_led_radius_map  = led_r_map*birght_field_led_mask

        led_r           = int(np.max(bright_field_led_radius_map)) * self.led_pitch
        illuminationNA  = led_r/math.sqrt(self.led_d_z**2+led_r**2)
        return illuminationNA
        
    def get_reconstruction_size(self) -> int:
        synthetic_NA            = self.NA_illu + self.NA
        reconstruct_dia_number  = math.ceil(2*synthetic_NA/self.wave_lambda/self.fourier_res)
        return reconstruct_dia_number       

    def get_led_r_angle_map(self) -> np.ndarray:
        led_ra_size         = math.floor(self.led_dia_number/2)
        led_h_idx_array     = np.linspace(-led_ra_size,led_ra_size,self.led_dia_number)
        led_v_idx_array     = np.linspace(-led_ra_size,led_ra_size,self.led_dia_number)
        
        led_h_idx_map, led_v_idx_map = np.meshgrid(led_h_idx_array,led_v_idx_array)
        led_r_map, led_angle_map     = cart2pol(led_h_idx_map,led_v_idx_map)
        return led_r_map, -led_angle_map

    def get_led_mask(self) -> np.ndarray:
        led_ra_size         = math.floor(self.led_dia_number/2)
        led_h_idx_array     = np.linspace(-led_ra_size,led_ra_size,self.led_dia_number)
        led_v_idx_array     = np.linspace(-led_ra_size,led_ra_size,self.led_dia_number)
        
        led_h_idx_map, led_v_idx_map = np.meshgrid(led_h_idx_array,led_v_idx_array)
        led_r_map, _                 = cart2pol(led_h_idx_map,led_v_idx_map)
        led_mask                     = led_r_map < self.led_dia_number/2
        return led_mask

    def get_led_index_map(self) -> np.ndarray:
        led_mask        = self.get_led_mask()
        led_index_map   = np.ones((self.led_dia_number,self.led_dia_number))
        led_index_map   = led_index_map * led_mask
        led_index_map   = led_index_map.reshape(self.led_dia_number**2,)
        led_idx = 1
        for idx in range(self.led_dia_number**2):
            if led_index_map[idx] != 0:
                led_index_map[idx] = led_idx
                led_idx += 1
        led_index_map   = led_index_map.reshape(self.led_dia_number,self.led_dia_number)
        return led_index_map

    def get_shiftsMap_shiftsPairs(self, return_sinThetaMap_ledMap:bool = False):
        led_ra_size         = math.floor(self.led_dia_number/2)
        led_h_idx_array     = np.linspace(-led_ra_size,led_ra_size,self.led_dia_number)
        led_v_idx_array     = np.linspace(-led_ra_size,led_ra_size,self.led_dia_number)
        
        led_h_idx_map, led_v_idx_map = np.meshgrid(led_h_idx_array,led_v_idx_array)
        led_r_map, _                 = cart2pol(led_h_idx_map,led_v_idx_map)
        led_r_map[led_r_map < self.led_dia_number/2] = 1
        led_r_map[led_r_map > self.led_dia_number/2] = 0

        n_used_led              = int(np.sum(led_r_map))

        led_h_d_map             = led_h_idx_map * self.led_pitch
        led_v_d_map             = led_v_idx_map * self.led_pitch
        led_to_sample_dist_map  = np.sqrt( (led_h_d_map)**2 + (led_v_d_map)**2 + self.led_d_z**2)

        sinTheta_h_map      = led_h_d_map/led_to_sample_dist_map
        sinTheta_v_map      = led_v_d_map/led_to_sample_dist_map

        shifts_h_map = np.round(sinTheta_h_map * led_r_map / self.wave_lambda / self.fourier_res)
        shifts_v_map = np.round(sinTheta_v_map * led_r_map / self.wave_lambda / self.fourier_res)

        shifts_h    = shifts_h_map[led_r_map == 1]
        shifts_v    = shifts_v_map[led_r_map == 1]
        shifts_pair = np.concatenate([shifts_v.reshape(n_used_led,1),shifts_h.reshape(n_used_led,1)],axis=1)

        if return_sinThetaMap_ledMap:
            return sinTheta_h_map, sinTheta_v_map, led_r_map
        else:
            return shifts_h_map, shifts_v_map, shifts_pair

    def get_bright_field_LED_map(self):
        sinTheta_h_map, sinTheta_v_map, _ = self.get_shiftsMap_shiftsPairs(return_sinThetaMap_ledMap = True)
        
        led_NA_map      = np.sqrt(sinTheta_h_map**2 + sinTheta_v_map**2)
        led_index_map   = self.get_led_index_map()

        birght_field_led_mask = (led_NA_map <= self.NA)
        bright_field_led_map = (birght_field_led_mask * led_index_map).astype(int)

        return bright_field_led_map, birght_field_led_mask
    
    def get_bright_field_multiplex_led_array(self, angle_range:np.ndarray, show_angle_map:bool= False) -> np.ndarray:
        led_r_map, led_angle_map = self.get_led_r_angle_map()
        led_index_map            = self.get_led_index_map()
        led_mask                 = self.get_led_mask()
        _, birght_field_led_mask = self.get_bright_field_LED_map()

        led_360_anlge_map = np.copy(led_angle_map)
        led_360_anlge_map = np.flip(np.abs((led_360_anlge_map - np.abs(led_360_anlge_map))/2), axis=1)
        led_360_anlge_map[9,:] = 0
        led_angle_map[led_angle_map<-1] = 180
        led_angle_map     = led_angle_map + led_360_anlge_map

        multiplex_led_list = []
        for idx in range(angle_range.shape[0]):
            if (angle_range[idx,1] - angle_range[idx,0])<0:
                led_angle_mask           = np.logical_and(np.logical_or(angle_range[idx,0]<led_angle_map,led_angle_map<angle_range[idx,1]),led_r_map!=0)
            else:
                led_angle_mask           = np.logical_and(angle_range[idx,0]<led_angle_map,led_angle_map<angle_range[idx,1])
            multiplex_led_idx_map    = led_index_map*led_mask*birght_field_led_mask*led_angle_mask
            multiplex_led_list.append(multiplex_led_idx_map[led_mask]>1)
            if show_angle_map:
                plt.figure()
                plt.imshow(multiplex_led_idx_map>0, cmap=cm.Greys_r)
                plt.colorbar()
                plt.title(f'Lit LED pattern {idx}')
                plt.savefig(f'led_pattern/led_map{idx}.png')
                plt.close()
        multiplex_led_array      = np.array(multiplex_led_list)
        return multiplex_led_array

    def get_single_dark_multiplex_led_array_by_radius_and_angle(self, single_angle_range:list, singe_radius_range:list) -> np.ndarray:
        # Convert led_angle_map to 0-360 degree map
        led_r_map, led_angle_map        = self.get_led_r_angle_map()
        led_360_anlge_map               = np.copy(led_angle_map)
        led_360_anlge_map               = np.flip(np.abs((led_360_anlge_map - np.abs(led_360_anlge_map))/2), axis=1)
        led_360_anlge_map[9,:]          = 0
        led_angle_map[led_angle_map<-1] = 180
        led_angle_map                   = led_angle_map + led_360_anlge_map

        dark_field_radius_range_led_mask           = np.logical_and(singe_radius_range[0]<=led_r_map,led_r_map<=singe_radius_range[1])
        _, birght_field_led_mask                   = self.get_bright_field_LED_map()
        dark_field_radius_range_led_mask[birght_field_led_mask] = False

        if (single_angle_range[1] - single_angle_range[0])<0:
            dark_field_angle_range_led_mask  = np.logical_and(np.logical_or(single_angle_range[0]<led_angle_map,led_angle_map<single_angle_range[1]),led_r_map!=0)
        else:
            dark_field_angle_range_led_mask  = np.logical_and(single_angle_range[0]<led_angle_map,led_angle_map<single_angle_range[1])
        
        led_index_map            = self.get_led_index_map()
        led_mask                 = self.get_led_mask()
        multiplex_led_idx_map    = led_index_map*led_mask*dark_field_radius_range_led_mask*dark_field_angle_range_led_mask

        multiplex_led_array      = np.array(multiplex_led_idx_map[led_mask])

        return multiplex_led_array, multiplex_led_idx_map

    def get_multiplex_led_array(self, multi_angle_range:list, multi_radius_range:list, show_angle_map:bool= False)-> np.ndarray:
        multiplex_led_list = []

        for idx_lists, angle_lists in enumerate(multi_angle_range):
            multiplex_led_idx_map = None
            multiplex_led_array   = None

            if isinstance(angle_lists[0],list):
                for idx_list, angle_list in enumerate(angle_lists):
                    _multiplex_led_array, _multiplex_led_idx_map = self.get_single_dark_multiplex_led_array_by_radius_and_angle(
                        single_angle_range= angle_list, singe_radius_range= multi_radius_range[idx_lists][idx_list])

                    if multiplex_led_idx_map is None:
                        multiplex_led_idx_map = _multiplex_led_idx_map
                    else:
                        multiplex_led_idx_map += _multiplex_led_idx_map

                    if multiplex_led_array is None:
                        multiplex_led_array = _multiplex_led_array
                    else:
                        multiplex_led_array += _multiplex_led_array

            else:
                _multiplex_led_array, _multiplex_led_idx_map = self.get_single_dark_multiplex_led_array_by_radius_and_angle(
                    single_angle_range= angle_lists, singe_radius_range= multi_radius_range[idx_lists])
                multiplex_led_idx_map = _multiplex_led_idx_map
                multiplex_led_array = _multiplex_led_array

            multiplex_led_list.append(multiplex_led_array>0)

            if show_angle_map:
                plt.figure()
                plt.imshow(multiplex_led_idx_map>0, cmap=cm.Greys_r)
                plt.colorbar()
                plt.title(f'Lit Dark LED pattern {idx_lists}')
                plt.savefig(f'led_pattern/dark_led_map{idx_lists}.png')
                plt.close()

        multiplex_led_array      = np.array(multiplex_led_list)
        return multiplex_led_array
    
    def get_multiplex_led_array_by_radius(self, angle_range:np.ndarray, radius:np.ndarray, show_angle_map:bool= False) -> np.ndarray:
        # Convert led_angle_map to 0-360 degree map
        led_r_map, led_angle_map = self.get_led_r_angle_map()
        led_360_anlge_map = np.copy(led_angle_map)
        led_360_anlge_map = np.flip(np.abs((led_360_anlge_map - np.abs(led_360_anlge_map))/2), axis=1)
        led_360_anlge_map[9,:] = 0
        led_angle_map[led_angle_map<-1] = 180
        led_angle_map     = led_angle_map + led_360_anlge_map

        led_index_map            = self.get_led_index_map()
        led_mask                 = self.get_led_mask()

        dark_field_led_mask      = np.logical_and(radius[0]<=led_r_map,led_r_map<=radius[1])
        _, birght_field_led_mask = self.get_bright_field_LED_map()

        multiplex_led_list = []
        total_overlap      = np.zeros_like(led_r_map)
        for idx in range(angle_range.shape[0]):
            if (angle_range[idx,1] - angle_range[idx,0])<0:
                led_angle_mask           = np.logical_and(np.logical_or(angle_range[idx,0]<led_angle_map,led_angle_map<angle_range[idx,1]),led_r_map!=0)
            else:
                led_angle_mask           = np.logical_and(angle_range[idx,0]<led_angle_map,led_angle_map<angle_range[idx,1])
            multiplex_led_idx_map    = led_index_map*led_mask*dark_field_led_mask*led_angle_mask
            multiplex_led_list.append(multiplex_led_idx_map[led_mask]>1)
            if show_angle_map:
                overlap_led_mask         = np.logical_and(multiplex_led_idx_map, birght_field_led_mask)
                _img = (multiplex_led_idx_map>0).astype(int) + overlap_led_mask.astype(int)
                plt.figure()
                plt.imshow(_img, cmap=cm.Greys_r)
                plt.colorbar()
                plt.title(f'Lit Dark Overlap LED pattern {idx}')
                plt.savefig(f'led_pattern/dark_overlap_led_map{idx}.png')
                plt.close()
                total_overlap += _img

                plt.figure()
                plt.imshow(multiplex_led_idx_map>0, cmap=cm.Greys_r)
                plt.colorbar()
                plt.title(f'Lit Dark LED pattern {idx}')
                plt.savefig(f'led_pattern/dark_led_map{idx}.png')
                plt.close()

        if show_angle_map:
            plt.figure()
            plt.imshow(total_overlap, cmap=cm.Greys_r)
            plt.colorbar()
            plt.title(f'Total Lit Dark Overlap LED pattern')
            plt.savefig(f'led_pattern/total_dark_overlap_led_map.png')
            plt.close()

        multiplex_led_array      = np.array(multiplex_led_list)

        led_r           = int(np.max(radius)) * self.led_pitch
        illuminationNA  = led_r/math.sqrt(self.led_d_z**2+led_r**2)

        self.NA_illu         = illuminationNA
        reconstruction_size  = self.get_reconstruction_size()
        reconstruction_shape = (reconstruction_size, reconstruction_size)

        return multiplex_led_array, reconstruction_shape

    def multiplex_image_select_by_array(self, multiplex_led_array:np.ndarray):      
        h_start     = int(self.CAMERA_H_RES//2  - self.camera_size//2)
        v_start     = int(self.CAMERA_V_RES//2  - self.camera_size//2)
        crop_size   = self.camera_size
        
        img_idx_array = np.linspace(1,293,293).astype(int)
        y = None
        print('image loading...')
        for _, i_img_mask in enumerate(multiplex_led_array):
            single_img = np.zeros_like(self.pupil_mask)
            for i in img_idx_array[i_img_mask]:
                file_name = str('ILED_{0:04}.tif'.format(i))
                img       = plt.imread(DAT_FILE_PATH + file_name)

                crop_img = img[v_start:v_start+crop_size, h_start:h_start+crop_size]
                single_img += crop_img
            
            if y is None:
                y = np.uint64(single_img)
            else:
                y = np.concatenate([y,np.uint64(single_img)], axis=0)
        print('finish loading...')  

        return y 
    
    def select_image(self, img_index_array, centre:list = [0, 0], load_img_or_not:bool = True, remove_background:bool = False, show_background:bool = False):
        shifts_pair     = self.total_shifts_pair[(img_index_array-1),0:2]

        h_start     = int(self.CAMERA_H_RES//2 + centre[0] - self.camera_size//2)
        v_start     = int(self.CAMERA_V_RES//2 - centre[1] - self.camera_size//2)
        crop_size   = self.camera_size
        if load_img_or_not:
            print('image loading...')
            img_list = []
            for i in img_index_array:
                file_name = str('ILED_{0:04}.tif'.format(i))
                img       = plt.imread(DAT_FILE_PATH + file_name)

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
                _crop_img_demean = _crop_img - img_background 
                if y is None:
                    y = _crop_img_demean
                else:
                    y = np.concatenate([y,_crop_img_demean], axis=0)
            
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
        plt.close()

    def total_shifts_rendering(self) -> None:
        plt.figure()
        plt.imshow(self.total_shifts_h_map, cmap=cm.Greys_r)
        plt.colorbar()
        plt.title('Total Horizental shifts map')
        plt.close()

        plt.figure()
        plt.imshow(self.total_shifts_v_map, cmap=cm.Greys_r)
        plt.colorbar()
        plt.title('Total Vertical shifts map')
        plt.close()
## ================================================ END Dataset ===================================================
## ================================================================================================================



## ================================================== Test Class ==================================================
## ================================================================================================================
class DPC_test(object):
    def __init__(self) -> None:
        pass

    def FPM_img(self, camera_size, n_iter= 1, linear_n_iter= 5,lr= 1):
        print('REAL dataset test \n----------------------')
        ## 1. use experimental setup from Laura dataset
        self.ptycho_data    = ptycho2d_Laura_dataSet(camera_size)
        bright_LED_map,_    = self.ptycho_data.get_bright_field_LED_map()
        bright_LED_array    = bright_LED_map[bright_LED_map>0] 

        ## 2. ground truth x generating
        reconstruction_res          = self.ptycho_data.get_reconstruction_size()
        reconstruction_shape        = (reconstruction_res,reconstruction_res)
        print(f'recontruction size: {reconstruction_shape}')

        ## 4. ptycho2d model create
        img_idx_array               = bright_LED_array
        probe                       = cp.array(self.ptycho_data.get_pupil_mask())
        y, img_list, shifts_pair    = self.ptycho_data.select_image(img_idx_array, remove_background= False)
        y                           = cp.array(y)
        self.ptycho_2d_model        = phaseretrieval.FourierPtychography2d(probe= probe, shifts_pair= shifts_pair, reconstruct_shape= reconstruction_shape)

        sum_bright_img = np.uint64(np.zeros_like(img_list[0]))
        for idx in range(len(img_list)):
            sum_bright_img += img_list[idx]

        ## 5. PPR solver
        initial_est             = cp.ones(shape=(reconstruction_res,reconstruction_res), dtype=np.complex128)
        initial_est             = np.fft.fft2(initial_est, norm="ortho")

        ppr_method              = algos.PerturbativePhase(self.ptycho_2d_model)
        x_est                   = ppr_method.iterate_GradientDescent(y = y, initial_est = initial_est, n_iter = n_iter, linear_n_iter= linear_n_iter, lr = lr)
        # x_est                   = ppr_method.iterate_ConjugateGradientDescent(y= y, initial_est= initial_est, n_iter= n_iter, linear_n_iter= linear_n_iter)
        # loss_function           = loss.loss_amplitude_based(epsilon=1)
        # gd_method               = algos.GradientDescent(self.ptycho_2d_model, loss_func= None, line_search= True)
        # x_est                   = gd_method.iterate(y = y, initial_est = initial_est, n_iter = n_iter, lr = lr)

        x_est                   = np.fft.ifft2(x_est, norm="ortho")

        ## 7. result
        plt.figure()
        plt.imshow(sum_bright_img, cmap=cm.Greys_r)
        plt.colorbar()
        plt.title('Sum Bright Field Image')
        plt.savefig('_recon_img/FPM_Sum Bright Field Image.png')
        plt.close()

        plt.figure()
        plt.imshow(img_list[int(len(img_list)/2)-1], cmap=cm.Greys_r)
        plt.colorbar()
        plt.title('Center Bright Field Image')
        plt.savefig('_recon_img/FPM_Center Bright Field Image.png')
        plt.close()

        plt.figure()
        plt.imshow(np.abs(x_est.get())**2, cmap=cm.Greys_r)
        plt.colorbar()
        plt.title('Intensity: Reconstructed image')
        plt.savefig('_recon_img/FPM_Intensity: Reconstructed image.png')
        plt.close()

        plt.figure()
        plt.imshow(np.angle(x_est.get()), cmap=cm.Greys_r)
        plt.colorbar()
        plt.title('Phase: Reconstruction image')
        plt.savefig('_recon_img/FPM_Phase: Reconstruction image.png')
        plt.close()
        return x_est
    
    def DPC_img(self, camera_size, angle_range:np.ndarray, n_iter= 1, GD_n_iter= 5,lr= 1):
        print('REAL dataset test \n----------------------')
        ## 1. use experimental setup from Laura dataset
        camera_size         = camera_size
        self.ptycho_data    = ptycho2d_Laura_dataSet(camera_size)

        ## 2. ground truth x generating
        # reconstruction_res          = camera_size
        reconstruction_res          = self.ptycho_data.get_reconstruction_size()
        reconstruction_shape        = (reconstruction_res, reconstruction_res)
        print(f'DPC recontruction shape: {reconstruction_shape}')

        ## 3. LED pattern select
        angle_range                     = angle_range
        multiplex_led_array             = self.ptycho_data.get_bright_field_multiplex_led_array(angle_range= angle_range, show_angle_map= True)
        self.DPC_multiplex_led_array    = multiplex_led_array
        y                               = self.ptycho_data.multiplex_image_select_by_array(multiplex_led_array= multiplex_led_array)
        y                               = cp.array(y)
        self.DPC_y                      = y

        ## 4. ptycho2d model create
        probe                       = cp.array(self.ptycho_data.get_pupil_mask())
        total_shifts_pair           = self.ptycho_data.total_shifts_pair

        ## 5. initial guess
        initial_est             = cp.ones(shape=(reconstruction_res,reconstruction_res), dtype=np.complex128)
        initial_est             = np.fft.fft2(initial_est, norm="ortho")

        pr_model                = phaseretrieval.MultiplexedPhaseRetrieval(probe= probe,multiplex_led_mask= multiplex_led_array, shifts_pair= total_shifts_pair, reconstruct_shape= reconstruction_shape)

        ppr_method              = algos.PerturbativePhase(pr_model)
        # x_est                   = ppr_method.iterate_GradientDescent(y= y, initial_est= initial_est, n_iter= n_iter, linear_n_iter= GD_n_iter, lr=lr)
        x_est                   = ppr_method.iterate_ConjugateGradientDescent(y= y, initial_est= initial_est, n_iter= n_iter, linear_n_iter= GD_n_iter)
        x_est                   = np.fft.ifft2(x_est, norm="ortho")

        plt.figure()
        plt.imshow(np.abs(x_est.get())**2, cmap=cm.Greys_r)
        plt.colorbar()
        plt.title('Intensity: Reconstructed image')
        plt.savefig('_recon_img/DPC_Intensity: Reconstructed image.png')
        plt.close()

        plt.figure()
        plt.imshow(np.angle(x_est.get()), cmap=cm.Greys_r)
        plt.colorbar()
        plt.title('Phase: Reconstruction image')
        plt.savefig('_recon_img/DPC_Phase: Reconstruction image.png')
        plt.close()

        plt.figure()
        plt.imshow(np.log(np.abs(np.fft.fftshift(np.fft.fft2(np.angle(x_est.get()), norm="ortho")))), cmap=cm.Greys_r)
        plt.colorbar()
        plt.title('Log DPC FFT')
        plt.savefig('_recon_img/DPC FFT image.png')
        plt.close()

        return x_est
    
    def dark_field_with_DPCimg(self, camera_size, bright_angle_range:np.ndarray, dark_angle_range:np.ndarray, dark_radius,
            bright_n_iter= 1, bright_linear_n_iter= 5, bright_lr= 1,
            dark_n_iter= 100, dark_linear_n_iter= 5, dark_lr= 1,
            dark_reconstruction:bool= False):
        print('Dark Field test \n----------------------')

        initial_est = self.DPC_img(camera_size= camera_size, angle_range= bright_angle_range, n_iter= bright_n_iter, GD_n_iter= bright_linear_n_iter,lr= bright_lr)
        initial_est = np.fft.fft2(initial_est, norm="ortho")
        
        
        dark_multiplex_led_array, dark_reconstruction_shape             = self.ptycho_data.get_multiplex_led_array_by_radius(angle_range= dark_angle_range, radius= dark_radius, show_angle_map= True)
        dark_y                      = self.ptycho_data.multiplex_image_select_by_array(multiplex_led_array= dark_multiplex_led_array)
        dark_y                      = cp.array(dark_y)

        probe                       = cp.array(self.ptycho_data.get_pupil_mask())
        total_shifts_pair           = self.ptycho_data.total_shifts_pair

        if dark_reconstruction:
            reconstruction_shape    = dark_reconstruction_shape
            initial_est_angle       = interpolate_img(np.angle(initial_est), reconstruction_shape[0])
            initial_est_amp         = interpolate_img(np.abs(initial_est), reconstruction_shape[0])
            initial_est             = cp.array(P2R(initial_est_amp,initial_est_angle)).astype(cp.complex128)
        else:
            reconstruction_shape   = initial_est.shape
        print(f'Dark recontruction shape: {reconstruction_shape}')

        total_y                 = np.concatenate((self.DPC_y, dark_y), axis=0)
        total_multiplex_led_mask= np.concatenate((self.DPC_multiplex_led_array,dark_multiplex_led_array), axis=0)
        pr_model                = phaseretrieval.MultiplexedPhaseRetrieval(probe= probe,multiplex_led_mask= total_multiplex_led_mask, shifts_pair= total_shifts_pair, reconstruct_shape= reconstruction_shape)
        ppr_method              = algos.PerturbativePhase(pr_model)

        # x_est                   = ppr_method.iterate_GradientDescent(y= y, initial_est= initial_est, n_iter= dark_n_iter, linear_n_iter= dark_linear_n_iter, lr= dark_lr)
        x_est                   = ppr_method.iterate_ConjugateGradientDescent(y= total_y, initial_est= initial_est, n_iter= dark_n_iter, linear_n_iter= dark_linear_n_iter)
        x_est                   = np.fft.ifft2(x_est, norm="ortho")

        plt.figure()
        plt.imshow(np.abs(x_est.get())**2, cmap=cm.Greys_r)
        plt.colorbar()
        plt.title('Intensity: Reconstructed image')
        plt.savefig('_recon_img/Intensity: Dark_field_DPC Reconstructed image.png')

        plt.figure()
        plt.imshow(np.angle(x_est.get()), cmap=cm.Greys_r)
        plt.colorbar()
        plt.title('Phase: Reconstruction image')
        plt.savefig('_recon_img/Phase: Dark_field_DPC Reconstruction image.png')

        plt.figure()
        plt.imshow(np.log(np.abs(np.fft.fftshift(np.fft.fft2(np.angle(x_est.get()), norm="ortho")))), cmap=cm.Greys_r)
        plt.colorbar()
        plt.title('Log Dark field+DPC FFT')
        plt.savefig('_recon_img/FFT: Dark_field_DPC FFT image.png')

        # plt.figure()
        # plt.imshow((np.abs(np.fft.fftshift(np.fft.fft2(np.angle(x_est.get()), norm="ortho"))) - np.abs(np.fft.fftshift(np.fft.fft2(np.angle(initial_est.get()), norm="ortho"))))>0, cmap=cm.Greys_r)
        # plt.colorbar()
        # plt.title('FFT Enhancement')
        # plt.savefig('_recon_img/FFT Enhancement.png')

        # plt.figure()
        # plt.imshow((np.abs(np.fft.fftshift(np.fft.fft2(np.angle(x_est.get()), norm="ortho"))) - np.abs(np.fft.fftshift(np.fft.fft2(np.angle(initial_est.get()), norm="ortho"))))<0, cmap=cm.Greys_r)
        # plt.colorbar()
        # plt.title('FFT Loss')
        # plt.savefig('_recon_img/FFT Loss.png')

        plt.close('all')

        return x_est


class DPC_image(object):
    def __init__(self,camera_size:int, angle_range:np.ndarray) -> None:
        self.camera_size = camera_size
        self.bright_field_angle_range = angle_range

        self.ptycho_data    = ptycho2d_Laura_dataSet(camera_size)

    def DPC_x_est(self, n_iter= 1, linear_n_iter= 5,lr= 1):
        print('DPC image solving \n----------------------')

        reconstruction_res          = self.ptycho_data.get_reconstruction_size()
        reconstruction_shape        = (reconstruction_res, reconstruction_res)
        print(f'DPC recontruction shape: {reconstruction_shape}')

        self.DPC_multiplex_led_array    = self.ptycho_data.get_bright_field_multiplex_led_array(angle_range= self.bright_field_angle_range, show_angle_map= True)
        self.y                          = self.ptycho_data.multiplex_image_select_by_array(multiplex_led_array= self.DPC_multiplex_led_array)
        self.y                          = cp.array(self.y)

        probe                       = cp.array(self.ptycho_data.get_pupil_mask())
        total_shifts_pair           = self.ptycho_data.total_shifts_pair

        initial_est             = cp.ones(shape=(reconstruction_res,reconstruction_res), dtype=np.complex128)
        initial_est             = np.fft.fft2(initial_est, norm="ortho")

        pr_model                = phaseretrieval.MultiplexedPhaseRetrieval(probe= probe,multiplex_led_mask= self.DPC_multiplex_led_array, shifts_pair= total_shifts_pair, reconstruct_shape= reconstruction_shape)

        ppr_method              = algos.PerturbativePhase(pr_model)

        if lr is not None:
            x_est                   = ppr_method.iterate_GradientDescent(y= self.y, initial_est= initial_est, n_iter= n_iter, linear_n_iter= linear_n_iter, lr=lr)
        else:
            x_est                   = ppr_method.iterate_ConjugateGradientDescent(y= self.y, initial_est= initial_est, n_iter= n_iter, linear_n_iter= linear_n_iter)

        x_est                   = np.fft.ifft2(x_est, norm="ortho")

        plt.figure()
        plt.imshow(np.abs(x_est.get())**2, cmap=cm.Greys_r)
        plt.colorbar()
        plt.title('Intensity: Reconstructed image')
        plt.savefig('_recon_img/DPC_Intensity: Reconstructed image.png')
        plt.close()

        plt.figure()
        plt.imshow(np.angle(x_est.get()), cmap=cm.Greys_r)
        plt.colorbar()
        plt.title('Phase: Reconstruction image')
        plt.savefig('_recon_img/DPC_Phase: Reconstruction image.png')
        plt.close()

        plt.figure()
        plt.imshow(np.log(np.abs(np.fft.fftshift(np.fft.fft2(np.angle(x_est.get()), norm="ortho")))), cmap=cm.Greys_r)
        plt.colorbar()
        plt.title('Log DPC FFT')
        plt.savefig('_recon_img/DPC FFT image.png')
        plt.close()

        return x_est
    
class dark_field_multiplex_with_DPC(object):
    def __init__(self,DPC_image_obj: DPC_image) -> None:
        self.DPC_image = DPC_image_obj

    def multiplex_x_est(self, multi_angle_range_list:list, multi_radius_range_list:list,
            bright_n_iter= 1, bright_linear_n_iter= 5, bright_lr= None,
            dark_n_iter= 100, dark_linear_n_iter= 5, dark_lr= None,
            dark_reconstruction:bool= False):

        initial_est = self.DPC_image.DPC_x_est(n_iter= bright_n_iter, linear_n_iter= bright_linear_n_iter, lr= bright_lr)
        initial_est = np.fft.fft2(initial_est, norm="ortho")
    
        reconstruction_shape   = initial_est.shape
        print(f'Dark recontruction shape: {reconstruction_shape}')
        
        dark_multiplex_led_array    = self.DPC_image.ptycho_data.get_multiplex_led_array(
            multi_angle_range= multi_angle_range_list, multi_radius_range= multi_radius_range_list,
            show_angle_map= True)
        dark_y                      = self.DPC_image.ptycho_data.multiplex_image_select_by_array(multiplex_led_array= dark_multiplex_led_array)
        dark_y                      = cp.array(dark_y)

        probe                       = cp.array(self.DPC_image.ptycho_data.get_pupil_mask())
        total_shifts_pair           = self.DPC_image.ptycho_data.total_shifts_pair

        total_y                     = np.concatenate((self.DPC_image.y, dark_y), axis=0)
        total_multiplex_led_mask    = np.concatenate((self.DPC_image.DPC_multiplex_led_array, dark_multiplex_led_array), axis=0)

        pr_model                    = phaseretrieval.MultiplexedPhaseRetrieval(probe= probe,multiplex_led_mask= total_multiplex_led_mask, shifts_pair= total_shifts_pair, reconstruct_shape= reconstruction_shape)
        ppr_method                  = algos.PerturbativePhase(pr_model)

        if dark_lr is not None:
            x_est               = ppr_method.iterate_GradientDescent(y= total_y, initial_est= initial_est, n_iter= dark_n_iter, linear_n_iter= dark_linear_n_iter, lr= dark_lr)
        else:
            x_est               = ppr_method.iterate_ConjugateGradientDescent(y= total_y, initial_est= initial_est, n_iter= dark_n_iter, linear_n_iter= dark_linear_n_iter)

        x_est                   = np.fft.ifft2(x_est, norm="ortho")

        plt.figure()
        plt.imshow(np.abs(x_est.get())**2, cmap=cm.Greys_r)
        plt.colorbar()
        plt.title('Intensity: Reconstructed image')
        plt.savefig('_recon_img/Intensity: Dark_field_DPC Reconstructed image.png')

        plt.figure()
        plt.imshow(np.angle(x_est.get()), cmap=cm.Greys_r)
        plt.colorbar()
        plt.title('Phase: Reconstruction image')
        plt.savefig('_recon_img/Phase: Dark_field_DPC Reconstruction image.png')

        plt.figure()
        plt.imshow(np.log(np.abs(np.fft.fftshift(np.fft.fft2(np.angle(x_est.get()), norm="ortho")))), cmap=cm.Greys_r)
        plt.colorbar()
        plt.title('Log Dark field+DPC FFT')
        plt.savefig('_recon_img/FFT: Dark_field_DPC FFT image.png')

        plt.close('all')

        return x_est


## ================================================== END Test Class ==================================================
## ====================================================================================================================


## ============ functions ============
def cart2pol(x, y):
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)/math.pi * 180
    return(rho, phi)

def interpolate_img(img, high_res:int):
    img_size         = img.shape[0]
    
    x_idx_array      = np.arange(-math.floor(img_size/2),math.ceil(img_size/2),1)
    high_x_idx_array = np.arange(-math.floor(img_size/2),math.ceil(img_size/2),img_size/high_res)

    inter_img_f      = interpolate.interp2d(x_idx_array, x_idx_array, img.get(), kind='linear')
    inter_img        = inter_img_f(high_x_idx_array, high_x_idx_array)

    return inter_img

def P2R(radii, angles):
    return radii * np.exp(1j*angles)
## ============ functions ============



if __name__ == '__main__':
    # dataset = ptycho2d_Laura_dataSet(256)
    # single_angle_range = np.array([0,180])
    # inner_dark_radius  = 2.5
    # outer_dark_radius  = 4.5
    # dark_radius        = np.array([np.sqrt(2*(inner_dark_radius**2)), np.sqrt(2*(outer_dark_radius**2))])
    # dataset.get_single_dark_multiplex_led_array_by_radius_and_angle(
    #     single_angle_range= single_angle_range, singe_radius_range= dark_radius, show_angle_map= True)

    # multi_angle_range_list  = [[[0,90],[180,270]], [90,180]]
    # multi_radius_range_list = [[[0,6],[0,9]],      [5,8]]

    # m_array=dataset.get_multiplex_led_array(
    #     multi_angle_range=multi_angle_range_list, multi_radius_range=multi_radius_range_list,
    #     show_angle_map= True)
    # print(m_array)
    # dataset.multiplex_image_select_by_array(multiplex_led_array= m_array)


    # ====================================================================================================
    DPC_obj = DPC_test()
    # 1. FPM No multiplex
    # DPC_obj.FPM_img(camera_size= 256, n_iter= 2, linear_n_iter= 50,lr= 1e-6)
    # DPC_obj.FPM_img(camera_size= 256, n_iter= 1, GD_n_iter= 5,lr= 1e-4)

    ## 2. DPC multiplex
    # angle_range = np.array([[0,180],[180,360]])
    # angle_range = np.array([[270,90],[90,270]])
    # angle_range = np.array([[0,180],[180,360],[270,90],[90,270]])
    # angle_range = np.array([[0,90],[90,180],[180,270],[270,360]])
    # DPC_obj.DPC_img(camera_size= 256, angle_range= angle_range, n_iter= 1, GD_n_iter= 5,lr= 1e-4)    

    ## 3. Dark field exploration with DPC img
    bright_angle_range = np.array([[0,180],[180,360]])
    # bright_angle_range = np.array([[270,90],[90,270]])
    # bright_angle_range = np.array([[0,180],[180,360],[270,90],[90,270]])
    # bright_angle_range = np.array([[0,90],[90,180],[180,270],[270,360]])
    DPCimg = DPC_image(camera_size= 256, angle_range= bright_angle_range)
    
    dark_field_multiplex_obj = dark_field_multiplex_with_DPC(DPC_image_obj= DPCimg)

    inner_dark_radius   = 2.5
    outer_dark_radius   = 4.5
    single_radius_range = [np.sqrt(2*(inner_dark_radius**2)), np.sqrt(2*(outer_dark_radius**2))]

    # multi_angle_range_list  = [[0,180],[180,360],[270,90],[90,270]]
    # multi_angle_range_list  = [[315,135],[135,315],[45,225],[225,45]]
    # multi_angle_range_list  = [[315,45],[45,135],[135,225],[225,315]]
    # multi_radius_range_list = [single_radius_range,single_radius_range,single_radius_range,single_radius_range]

    multi_angle_range_list  = [[[315,45],[135,180]],                           [[45,90],[270,315]], [[45,135],[180,270]]]
    multi_radius_range_list = [[single_radius_range,single_radius_range],      [[6.5,9],[6.5,9]],   [[9,12],single_radius_range]]

    dark_field_multiplex_obj.multiplex_x_est(
        multi_angle_range_list=multi_angle_range_list, multi_radius_range_list=multi_radius_range_list,
        bright_n_iter= 1, bright_linear_n_iter= 5, bright_lr= None,
        dark_n_iter= 3, dark_linear_n_iter= 25, dark_lr= None,
        dark_reconstruction= False)
    
    


    # dark_angle_range   = np.array([[0,180],[180,360]])
    # dark_angle_range   = np.array([[270,90],[90,270]])
    # dark_angle_range = np.array([[0,180],[180,360],[270,90],[90,270]])
    # dark_angle_range = np.array([[0,90],[90,180],[180,270],[270,360]])
    # inner_dark_radius  = 0
    # outer_dark_radius  = 4.5
    # dark_radius        = np.array([np.sqrt(2*(inner_dark_radius**2)), np.sqrt(2*(outer_dark_radius**2))])

    # DPC_obj.dark_field_with_DPCimg(camera_size=256,
    #     bright_angle_range= bright_angle_range, dark_angle_range= dark_angle_range, dark_radius= dark_radius,
        # bright_n_iter= 1, bright_linear_n_iter= 5, bright_lr= None,
        # dark_n_iter= 3, dark_linear_n_iter= 25, dark_lr= None,
        # dark_reconstruction= False)