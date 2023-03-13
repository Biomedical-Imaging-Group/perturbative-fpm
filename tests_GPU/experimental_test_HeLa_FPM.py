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

DAT_FILE_PATH = '/home/kshen/Ptychography_Project/phase-retrieval-library/phase-retrieval-library/dataset/DPC_dataset/' ## Local PATH


## ================================================== Dataset  ====================================================
## ================================================================================================================
class HeLa_cell_dataSet(object):
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
    def get_reconstruction_size(self, bright_field_NA:bool = False, multiplex_led_array_mask:np.ndarray = None) -> int:
        if bright_field_NA:
            NA_illu = self.get_bright_illumnationNA()
        elif multiplex_led_array_mask is not None:
            NA_illu = self.get_illumnationNA_by_mulitplex_array(multiplex_led_array_mask= multiplex_led_array_mask)
        else:
             NA_illu = self.get_total_illumnationNA()
        synthetic_NA            = NA_illu + self.NA
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

    def get_illumnationNA_by_mulitplex_array(self, multiplex_led_array_mask:np.ndarray) -> float:
        led_r_map,_         = self.get_led_r_angle_map()
        led_bool_mask       = self.get_led_bool_mask()
        led_r_array         = led_r_map[led_bool_mask]

        illuminationNA = 0
        for i_multiplex_array in multiplex_led_array_mask:
            temp_led_r = int(np.max(led_r_array[(i_multiplex_array>0)])) * self.led_pitch
            temp_illNA = temp_led_r/math.sqrt(self.led_d_z**2+temp_led_r**2)
            if temp_illNA > illuminationNA:
                illuminationNA = temp_illNA
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
            shifts_pair[145,:] = [0,0]
            # shifts_pair[146,:] = [0,25]

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
    
    def get_bright_field_multiplex_led_array_mask(self, angle_range:np.ndarray, show_angle_map:bool= False) -> np.ndarray:
        led_r_map, _                = self.get_led_r_angle_map()
        led_angle_map               = self.get_led_360_coordinate_angle_map()
        led_index_map               = self.get_led_index_map()
        led_bool_mask               = self.get_led_bool_mask()
        _, birght_field_led_mask, _ = self.get_bright_field_LED_map()

        multiplex_led_array_mask_list = []
        for idx in range(angle_range.shape[0]):
            if (angle_range[idx,1] - angle_range[idx,0])<0:
                led_angle_bool_mask           = np.logical_and(np.logical_or(angle_range[idx,0]<led_angle_map,led_angle_map<angle_range[idx,1]),led_r_map!=0)
            else:
                led_angle_bool_mask           = np.logical_and(angle_range[idx,0]<led_angle_map,led_angle_map<angle_range[idx,1])
            multiplex_led_idx_map           = led_index_map*led_bool_mask*birght_field_led_mask*led_angle_bool_mask

            multiplex_led_bool_array_mask   = multiplex_led_idx_map[led_bool_mask]>1
            multiplex_led_array_mask        = multiplex_led_bool_array_mask.astype(int)
            multiplex_led_array_mask_list.append(multiplex_led_array_mask)

            if show_angle_map:
                plt.figure()
                plt.imshow(multiplex_led_idx_map>0, cmap=cm.Greys_r)
                plt.colorbar()
                plt.title(f'Lit LED pattern {idx}')
                plt.savefig(f'led_pattern/led_map{idx}.png')
                plt.close()

        multiplex_led_array_mask                 = np.array(multiplex_led_array_mask_list)
        return multiplex_led_array_mask

    def get_single_dark_multiplex_led_array_mask(self, single_angle_range:list, single_radius_range:list) -> np.ndarray:
        # Convert led_angle_map to 0-360 degree map
        led_r_map, _                = self.get_led_r_angle_map()
        led_angle_map               = self.get_led_360_coordinate_angle_map()

        dark_field_radius_range_led_bool_mask                             = np.logical_and(single_radius_range[0]<=led_r_map,led_r_map<=single_radius_range[1])
        _, _,birght_field_led_bool_mask                                   = self.get_bright_field_LED_map()
        dark_field_radius_range_led_bool_mask[birght_field_led_bool_mask] = False

        if (single_angle_range[1] - single_angle_range[0])<0:
            dark_field_angle_range_led_bool_mask  = np.logical_and(np.logical_or(single_angle_range[0]<led_angle_map,led_angle_map<single_angle_range[1]),led_r_map!=0)
        else:
            dark_field_angle_range_led_bool_mask  = np.logical_and(single_angle_range[0]<led_angle_map,led_angle_map<single_angle_range[1])
        
        led_index_map            = self.get_led_index_map()
        led_bool_mask            = self.get_led_bool_mask()
        multiplex_led_idx_map    = led_index_map*led_bool_mask*dark_field_radius_range_led_bool_mask*dark_field_angle_range_led_bool_mask
        multiplex_led_idx_array  = np.array(multiplex_led_idx_map[led_bool_mask])
        multiplex_led_mask       = (multiplex_led_idx_map>0).astype(int)
        multiplex_led_array_mask = (multiplex_led_idx_array>0).astype(int)

        return multiplex_led_idx_map, multiplex_led_idx_array, multiplex_led_mask, multiplex_led_array_mask

    def get_dark_multiplex_led_array_mask(self, multi_angle_range:list, multi_radius_range:list, show_angle_map:bool= False)-> np.ndarray:
        multiplex_led_array_mask_list = []

        for idx_lists, angle_lists in enumerate(multi_angle_range):
            single_multiplex_led_array_mask   = None

            if isinstance(angle_lists[0],list):
                multiplex_led_mask = None
                for idx_list, angle_list in enumerate(angle_lists):
                    _, _, _multiplex_led_mask,_multiplex_led_array_mask = self.get_single_dark_multiplex_led_array_mask(
                        single_angle_range = angle_list, single_radius_range= multi_radius_range[idx_lists][idx_list])

                    if single_multiplex_led_array_mask is None:
                        single_multiplex_led_array_mask = _multiplex_led_array_mask
                    else:
                        single_multiplex_led_array_mask += _multiplex_led_array_mask

                    if multiplex_led_mask is None:
                        multiplex_led_mask  = _multiplex_led_mask
                    else:
                        multiplex_led_mask += _multiplex_led_mask
                    
            else:
                _, _, _multiplex_led_mask,_multiplex_led_array_mask = self.get_single_dark_multiplex_led_array_mask(
                    single_angle_range= angle_lists, single_radius_range= multi_radius_range[idx_lists])
                single_multiplex_led_array_mask = _multiplex_led_array_mask
                multiplex_led_mask              = _multiplex_led_mask

            multiplex_led_array_mask_list.append(single_multiplex_led_array_mask)

            if show_angle_map:
                plt.figure()
                plt.imshow(multiplex_led_mask, cmap=cm.Greys_r)
                plt.colorbar()
                plt.title(f'Lit Dark LED pattern {idx_lists}')
                plt.savefig(f'led_pattern/dark_led_map{idx_lists}.png')
                plt.close()

        multiplex_led_array_mask      = np.array(multiplex_led_array_mask_list)
        return multiplex_led_array_mask

    # ----------------------- Images selection methods ----------------------- 
    def select_image_by_multiplexArrayMask(self, multiplex_led_array_mask:np.ndarray, centre:list = [0, 0]):      
        print('image loading...')
        h_start     = int(self.CAMERA_H_RES//2 + centre[0] - self.camera_size//2)
        v_start     = int(self.CAMERA_V_RES//2 - centre[1] - self.camera_size//2)
        crop_size   = self.camera_size
        
        y = None
        for _, i_img_array_mask in enumerate(multiplex_led_array_mask):
            single_img = np.zeros_like(self.pupil_mask)
            for _idx, _weight in enumerate(i_img_array_mask):
                if _weight>0:
                    file_name = str('ILED_{0:04}.tif'.format(_idx+1))
                    img       = plt.imread(DAT_FILE_PATH + file_name)

                    crop_img    = img[v_start:v_start+crop_size, h_start:h_start+crop_size]*_weight
                    single_img += crop_img
            
            if y is None:
                y = np.uint64(single_img)
            else:
                y = np.concatenate([y,np.uint64(single_img)], axis=0)
        print('finish loading...')  

        return y 

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
        
    # ----------------------- Rendering ----------------------
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
class FPM_solver(object):
    def __init__(self) -> None:
        pass

    def FPM(self, camera_size, n_iter= 5, lr= 1, cropping_center=[0,0], amp_based_or_not:bool = True,for_loop_or_not:bool = False):
        print('Full FPM Start \n----------------------')
        
        self.dataset = HeLa_cell_dataSet(camera_size= camera_size)

        reconstruction_res          = self.dataset.get_reconstruction_size()
        reconstruction_shape        = (reconstruction_res,reconstruction_res)
        print(f'recontruction size: {reconstruction_shape}')

        img_idx_array               = np.linspace(1,293,293).astype(int)
        probe                       = cp.array(self.dataset.get_pupil_mask())
        y, img_list, shifts_pair    = self.dataset.select_image_by_imgIndex(img_index_array= img_idx_array, centre= cropping_center, remove_background= True)
        y                           = cp.array(y)
        self.phase_model            = phaseretrieval.FourierPtychography2d(probe= probe, shifts_pair= shifts_pair, reconstruct_shape= reconstruction_shape)

        initial_est                 = cp.ones(shape= reconstruction_shape, dtype=np.complex128)
        initial_est                 = np.fft.fft2(initial_est, norm="ortho")

        if amp_based_or_not:
            loss_function               = loss.loss_amplitude_based(epsilon= 1e-5)
            gd_method                   = algos.GradientDescent(self.phase_model, loss_func= loss_function, line_search= None)
        else:
            gd_method                   = algos.GradientDescent(self.phase_model, loss_func= None, line_search= True)
        x_est                       = gd_method.iterate(y = y, initial_est = initial_est, n_iter = n_iter, lr = lr)
        x_est                       = np.fft.ifft2(x_est, norm="ortho")

        if for_loop_or_not == False:
            plt.figure()
            plt.imshow(np.abs(x_est.get())**2, cmap=cm.Greys_r)
            plt.colorbar()
            plt.title('Intensity: Reconstructed image')
            plt.savefig('_recon_img/HeLa_FPM_Intensity: Reconstructed image.png')
            plt.close()

            plt.figure()
            plt.imshow(np.angle(x_est.get()), cmap=cm.Greys_r)
            plt.colorbar()
            plt.title('Phase: Reconstruction image')
            plt.savefig('_recon_img/HeLa_FPM_Phase: Reconstruction image.png')
            plt.close()

            sum_bright_img = np.uint64(np.zeros_like(img_list[0]))
            for idx in range(len(img_list)):
                sum_bright_img += img_list[idx]
            plt.figure()
            plt.imshow(sum_bright_img, cmap=cm.Greys_r)
            plt.colorbar()
            plt.title('Sum All Image')
            plt.savefig('_recon_img/HeLa_FPM_Sum All Image.png')
            plt.close()

            phase_img = np.angle(x_est)
            phase_img_99 = np.percentile(np.abs(phase_img),99.99)
            phase_img[phase_img > phase_img_99]  = 0
            phase_img[phase_img < -phase_img_99] = 0
            plt.figure()
            plt.imshow(phase_img.get(), cmap=cm.Greys_r)
            plt.colorbar()
            plt.title('Phase Image, Filter out 99.99%')
            plt.savefig('_recon_img/HeLa_FPM_Phase99: Reconstruction image.png')
            plt.close()

            histogram, bin_edges = np.histogram(np.angle(x_est.get()), bins= (x_est.shape[0]*x_est.shape[1]))
            plt.figure()
            plt.title("Phase Grayscale Histogram")
            plt.xlabel("grayscale value")
            plt.ylabel("pixel count")
            plt.plot(bin_edges[0:-1], histogram)
            _, max_ylim = plt.ylim()
            plt.text(phase_img_99.get()*1.1, max_ylim*0.9, '99.99%= {:.2f}'.format(phase_img_99.get()))
            plt.axvline(phase_img_99.get(), color='k', linestyle='dashed', linewidth=1)
            plt.savefig('_recon_img/HeLa_FPM_Phase Histogram.png')
            plt.close()

        else:
            file_path   = str('_FPM/amp_based')
            file_header = str('/HeLa_FPM_Bright_')
            file_iter   = str(f'n_iter={n_iter}, lr= {lr}.png')
            plt.figure()
            plt.imshow(np.abs(x_est.get())**2, cmap=cm.Greys_r)
            plt.colorbar()
            plt.title(f'Intensity: '+ file_iter)
            plt.savefig( file_path + f'/n_iter={n_iter}' + file_header + 'Intensity: ' + file_iter)
            plt.close()

            plt.figure()
            plt.imshow(np.angle(x_est.get()), cmap=cm.Greys_r)
            plt.colorbar()
            plt.title(f'Phase: '+ file_iter)
            plt.savefig( file_path + f'/n_iter={n_iter}' + file_header + 'Phase: ' + file_iter)
            plt.close()

            phase_img = np.angle(x_est)
            phase_img_99 = np.percentile(np.abs(phase_img),99.99)
            phase_img[phase_img > phase_img_99]  = 0
            phase_img[phase_img < -phase_img_99] = 0
            plt.figure()
            plt.imshow(phase_img.get(), cmap=cm.Greys_r)
            plt.colorbar()
            plt.title(f'Phase Image, Filter out 99.99%: ' + file_iter)
            plt.savefig( file_path + f'/n_iter={n_iter}' + file_header + 'Phase99: ' + file_iter)
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
            plt.savefig( file_path + f'/n_iter={n_iter}' + file_header + 'Phase Histogram: ' + file_iter)
            plt.close()
        return x_est
    
    def FPM_PPR(self, camera_size:int, n_iter:int, linear_n_iter, lr, centre:list= [0,0], img_idx_array:np.ndarray= None, for_loop_or_not:bool = False):
        print('FPM test \n----------------------')
        self.dataset                = HeLa_cell_dataSet(camera_size)

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
            plt.savefig('_recon_img/HeLa_FPM_Intensity image.png')

            plt.figure()
            plt.imshow(np.angle(x_est.get()), cmap=cm.Greys_r)
            plt.colorbar()
            plt.title('Phase: Reconstruction image')
            plt.savefig('_recon_img/HeLa_FPM__Phase image.png')
        else:
            file_path   = str(f'_FPM_PPR/GD/n_iter={n_iter}')
            file_header = str('/HeLa_FPM_')
            file_iter   = str(f'lr={lr}, linear_n_iter={linear_n_iter}.png')
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

            try:
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
            except:
                pass

        return x_est
    
    def bright_FPM(self, camera_size, n_iter= 5, lr= 1, cropping_center=[0,0], amp_based_or_not:bool = True,for_loop_or_not:bool = False):
        print('Bright field FPM Start \n----------------------')
        
        self.dataset = HeLa_cell_dataSet(camera_size= camera_size) 

        reconstruction_res          = self.dataset.get_reconstruction_size(bright_field_NA= True)
        reconstruction_shape        = (reconstruction_res,reconstruction_res)
        print(f'recontruction size: {reconstruction_shape}')

        bright_field_led_index_map, _, _ = self.dataset.get_bright_field_LED_map()
        bright_LED_index_array           = bright_field_led_index_map[bright_field_led_index_map>0]
        probe                       = cp.array(self.dataset.get_pupil_mask())
        y, img_list, shifts_pair    = self.dataset.select_image_by_imgIndex(img_index_array= bright_LED_index_array, centre= cropping_center, remove_background= True)
        y                           = cp.array(y)
        self.phase_model            = phaseretrieval.FourierPtychography2d(probe= probe, shifts_pair= shifts_pair, reconstruct_shape= reconstruction_shape)

        initial_est                 = cp.ones(shape= reconstruction_shape, dtype=np.complex128)
        initial_est                 = np.fft.fft2(initial_est, norm="ortho")

        if amp_based_or_not:
            loss_function               = loss.loss_amplitude_based(epsilon= 1e-1)
            gd_method                   = algos.GradientDescent(self.phase_model, loss_func= loss_function, line_search= None)
        else:
            gd_method                   = algos.GradientDescent(self.phase_model, loss_func= None, line_search= True)
        x_est                       = gd_method.iterate(y = y, initial_est = initial_est, n_iter = n_iter, lr = lr)
        x_est                       = np.fft.ifft2(x_est, norm="ortho")

        if for_loop_or_not == False:
            plt.figure()
            plt.imshow(np.abs(x_est.get())**2, cmap=cm.Greys_r)
            plt.colorbar()
            plt.title('Intensity image')
            plt.savefig('_recon_img/HeLa_FPM_Bright_Intensity image.png')
            plt.close()

            plt.figure()
            plt.imshow(np.angle(x_est.get()), cmap=cm.Greys_r)
            plt.colorbar()
            plt.title('Phase image')
            plt.savefig('_recon_img/HeLa_FPM_Bright_Phase image.png')
            plt.close()

            sum_bright_img = np.uint64(np.zeros_like(img_list[0]))
            for idx in range(len(img_list)):
                sum_bright_img += img_list[idx]
            plt.figure()
            plt.imshow(sum_bright_img, cmap=cm.Greys_r)
            plt.colorbar()
            plt.title('Sum Bright Field Image')
            plt.savefig('_recon_img/HeLa_FPM_Bright_Sum Bright Field Image.png')
            plt.close()

            phase_img = np.angle(x_est)
            phase_img_99 = np.percentile(np.abs(phase_img),99.995)
            phase_img[phase_img > phase_img_99]  = 0
            phase_img[phase_img < -phase_img_99] = 0
            plt.figure()
            plt.imshow(phase_img.get(), cmap=cm.Greys_r)
            plt.colorbar()
            plt.title('Phase Image, Filter out 99.995%')
            plt.savefig('_recon_img/HeLa_FPM_Bright_Phase99: Reconstruction image.png')
            plt.close()

            histogram, bin_edges = np.histogram(np.angle(x_est.get()), bins= (x_est.shape[0]*x_est.shape[1]))
            plt.figure()
            plt.title("Phase Grayscale Histogram")
            plt.xlabel("grayscale value")
            plt.ylabel("pixel count")
            plt.plot(bin_edges[0:-1], histogram)
            _, max_ylim = plt.ylim()
            plt.text(phase_img_99.get()*1.1, max_ylim*0.9, '99.995%= {:.2f}'.format(phase_img_99.get()))
            plt.axvline(phase_img_99.get(), color='k', linestyle='dashed', linewidth=1)
            plt.savefig('_recon_img/HeLa_FPM_Bright_Phase Histogram.png')
            plt.close()

        else:
            file_path   = str('_bright_FPM/amp_based')
            file_header = str('/HeLa_FPM_Bright_')
            file_iter   = str(f'n_iter={n_iter}, lr= {lr}.png')
            plt.figure()
            plt.imshow(np.abs(x_est.get())**2, cmap=cm.Greys_r)
            plt.colorbar()
            plt.title(f'Intensity: '+ file_iter)
            plt.savefig( file_path + f'/n_iter={n_iter}' + file_header + 'Intensity: ' + file_iter)
            plt.close()

            plt.figure()
            plt.imshow(np.angle(x_est.get()), cmap=cm.Greys_r)
            plt.colorbar()
            plt.title(f'Phase: '+ file_iter)
            plt.savefig( file_path + f'/n_iter={n_iter}' + file_header + 'Phase: ' + file_iter)
            plt.close()

            phase_img = np.angle(x_est)
            phase_img_99 = np.percentile(np.abs(phase_img),99.99)
            phase_img[phase_img > phase_img_99]  = 0
            phase_img[phase_img < -phase_img_99] = 0
            plt.figure()
            plt.imshow(phase_img.get(), cmap=cm.Greys_r)
            plt.colorbar()
            plt.title(f'Phase Image, Filter out 99.99%: ' + file_iter)
            plt.savefig( file_path + f'/n_iter={n_iter}' + file_header + 'Phase99: ' + file_iter)
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
            plt.savefig( file_path + f'/n_iter={n_iter}' + file_header + 'Phase Histogram: ' + file_iter)
            plt.close()
        return x_est

class DPC_and_darkField_solver(object):
    def __init__(self,camera_size:int) -> None:
        self.camera_size = camera_size
        self.dataset     = HeLa_cell_dataSet(camera_size= camera_size)

    def DPC(self, bright_field_angle_range:np.ndarray, n_iter= 1, linear_n_iter= 5, lr= 1, centre:list = [0,0], for_loop_or_not:bool = False):
        print('DPC start \n----------------------')

        reconstruction_res          = self.dataset.get_reconstruction_size(bright_field_NA= True)
        reconstruction_shape        = (reconstruction_res, reconstruction_res)
        print(f'DPC recontruction shape: {reconstruction_shape}')

        self.DPC_multiplex_led_array_mask   = self.dataset.get_bright_field_multiplex_led_array_mask(angle_range= bright_field_angle_range, show_angle_map= True)
        self.DPC_y                          = self.dataset.select_image_by_multiplexArrayMask(multiplex_led_array_mask= self.DPC_multiplex_led_array_mask, centre= centre)
        self.DPC_y                          = cp.array(self.DPC_y)

        probe                       = cp.array(self.dataset.get_pupil_mask())
        total_shifts_pair           = self.dataset.total_shifts_pair

        initial_est             = cp.ones(shape= reconstruction_shape, dtype=np.complex128)
        initial_est             = np.fft.fft2(initial_est, norm="ortho")

        pr_model                = phaseretrieval.MultiplexedPhaseRetrieval(probe= probe,multiplex_led_mask= self.DPC_multiplex_led_array_mask, shifts_pair= total_shifts_pair, reconstruct_shape= reconstruction_shape)
        ppr_method              = algos.PerturbativePhase(pr_model)

        if lr is not None:
            x_est                   = ppr_method.iterate_GradientDescent(y= self.DPC_y, initial_est= initial_est, n_iter= n_iter, linear_n_iter= linear_n_iter, lr=lr)
        else:
            x_est                   = ppr_method.iterate_ConjugateGradientDescent(y= self.DPC_y, initial_est= initial_est, n_iter= n_iter, linear_n_iter= linear_n_iter)

        x_est                   = np.fft.ifft2(x_est, norm="ortho")

        if for_loop_or_not == False:
            plt.figure()
            plt.imshow(np.abs(x_est.get())**2, cmap=cm.Greys_r)
            plt.colorbar()
            plt.title('Intensity: Reconstructed image')
            plt.savefig('_recon_img/HeLa_DPC_Intensity image.png')
            plt.close()

            plt.figure()
            plt.imshow(np.angle(x_est.get()), cmap=cm.Greys_r)
            plt.colorbar()
            plt.title('Phase: Reconstruction image')
            plt.savefig('_recon_img/HeLa_DPC_Phase image.png')
            plt.close()
        else:
            plt.figure()
            plt.imshow(np.abs(x_est.get())**2, cmap=cm.Greys_r)
            plt.colorbar()
            plt.title(f'Intensity GD: linear_n_iter={linear_n_iter}, n_iter={n_iter}, lr={lr}')
            plt.savefig(f'_dpc_img/top_bottom_right_left/GD/n_iter={n_iter}/Intensity GD n_iter={n_iter}, linear_n_iter={linear_n_iter}, lr={lr}.png')
            plt.close()

            plt.figure()
            plt.imshow(np.angle(x_est.get()), cmap=cm.Greys_r)
            plt.colorbar()
            plt.title(f'Phase GD: linear_n_iter={linear_n_iter}, n_iter={n_iter}, lr={lr}')
            plt.savefig(f'_dpc_img/top_bottom_right_left/GD/n_iter={n_iter}//Phase GD n_iter={n_iter}, linear_n_iter={linear_n_iter}, lr={lr}.png')
            plt.close()
        return x_est

    def dark_field_with_DPC(
            self, bright_field_angle_range:np.ndarray, dark_multi_angle_range_list:list, dark_multi_radius_range_list:list,
            bright_n_iter= 1, bright_linear_n_iter= 5, bright_lr= None,
            dark_n_iter= 100, dark_linear_n_iter= 5, dark_lr= None,
            centre:list= [0,0]):

        initial_est = self.DPC(bright_field_angle_range= bright_field_angle_range, n_iter= bright_n_iter, linear_n_iter= bright_linear_n_iter, lr= bright_lr, centre= centre)
        initial_est = np.fft.fft2(initial_est, norm="ortho")
        
        self.dark_multiplex_led_array_mask    = self.dataset.get_dark_multiplex_led_array_mask(
            multi_angle_range= dark_multi_angle_range_list, multi_radius_range= dark_multi_radius_range_list, show_angle_map= True)
        self.dark_y                           = self.dataset.select_image_by_multiplexArrayMask(multiplex_led_array_mask= self.dark_multiplex_led_array_mask, centre= centre)
        self.dark_y                           = cp.array(self.dark_y)

        probe                       = cp.array(self.dataset.get_pupil_mask())
        total_shifts_pair           = self.dataset.total_shifts_pair

        total_y                     = np.concatenate((self.DPC_y, self.dark_y), axis=0)
        total_multiplex_led_mask    = np.concatenate((self.DPC_multiplex_led_array_mask, self.dark_multiplex_led_array_mask), axis=0)

        reconstruction_size                = self.dataset.get_reconstruction_size(multiplex_led_array_mask= self.dark_multiplex_led_array_mask)
        reconstruction_shape               = (reconstruction_size, reconstruction_size)
        print(f'Dark recontruction shape: {reconstruction_shape}')

        crop_op     = LinOpCrop2(in_shape= reconstruction_shape, crop_shape= initial_est.shape)
        initial_est = np.fft.ifftshift(crop_op.applyT(np.fft.fftshift(initial_est)))

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
        plt.savefig('_recon_img/HeLa_Dark_and_DPC_Intensity image.png')

        plt.figure()
        plt.imshow(np.angle(x_est.get()), cmap=cm.Greys_r)
        plt.colorbar()
        plt.title('Phase: Reconstruction image')
        plt.savefig('_recon_img/HeLa_Dark_and_DPC_Phase image.png')
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

def delete_file(dir_name):
    dir_name = dir_name
    for file_name in os.listdir(dir_name):
        os.remove(os.path.join(dir_name, file_name))

def P2R(radii, angles):
    return radii * np.exp(1j*angles)
## ============ functions ============



if __name__ == '__main__':
    ## clean folder
    # delete_file('led_pattern')
    # delete_file('_recon_img')   
    # ====================================================================================================
    # ====================================================================================================
    ## 1. FPM
    FPM_test = FPM_solver()
    cropping_center = [0,0]
    # FPM_test.FPM(camera_size= 256, n_iter= 5, cropping_center= cropping_center, amp_based_or_not= False, lr= 1)
    # for n_iter in [1,2,3,4,5]:
    #     FPM_test.FPM(camera_size= 256, n_iter= n_iter, cropping_center= cropping_center, amp_based_or_not=False, lr= 1, for_loop_or_not= True)
    # for n_iter in [35]:
    #     for lr in np.geomspace(1e-2, 1e-2, num=1):
    #         FPM_test.FPM(camera_size= 256, n_iter= n_iter, cropping_center= cropping_center, amp_based_or_not=True, lr= lr, for_loop_or_not= True)

    # n_iter = 5
    # delete_file(f'_FPM_PPR/GD/n_iter={n_iter}') 
    # for lr in np.geomspace(1e-4, 1e-8, num=5):  
    #     for linear_n_iter in [1,3,5,7,9]:
    # FPM_test.FPM_PPR(camera_size= 256, n_iter= 1, linear_n_iter= 1, centre= cropping_center, lr= None, for_loop_or_not= False)


    # FPM_test.bright_FPM(camera_size= 128, n_iter= 50, cropping_center= cropping_center, amp_based_or_not=False, lr= 1)
    # for n_iter in [25,50]:
    #     for lr in np.geomspace(1e-2, 1e-5, num=4):
    #         FPM_test.bright_FPM(camera_size= 256, n_iter= n_iter, cropping_center= cropping_center, amp_based_or_not=True, lr= lr, for_loop_or_not= True)
    # for n_iter in [10,25,50,100]:
    #     FPM_test.bright_FPM(camera_size= 256, n_iter= n_iter, cropping_center= cropping_center, amp_based_or_not=False, lr= 1, for_loop_or_not= True)
    # ====================================================================================================
    # ====================================================================================================
    ## 2. DPC
    DPC_and_darkField_test = DPC_and_darkField_solver(camera_size= 256)

    # bright_angle_range = np.array([[0,90],[90,180],[180,270],[270,360]])
    # DPC_and_darkField_test.DPC(bright_field_angle_range= bright_angle_range, n_iter= 1, linear_n_iter= 3, lr= None, centre=[-125,300])
    # ====================================================================================================
    # ====================================================================================================
    ## 3. DPC and Dark Field
    DPC_and_darkField_test = DPC_and_darkField_solver(camera_size= 256)

    centre =[-125,300]

    bright_angle_range = np.array([[0,90],[90,180],[180,270],[270,360]])

    inner_dark_radius   = 2.5
    outer_dark_radius   = 4.5
    single_radius_range = [np.sqrt(2*(inner_dark_radius**2)), np.sqrt(2*(outer_dark_radius**2))]

    multi_angle_range_list  = [[0,180],[180,360],[270,90],[90,270]]
    multi_radius_range_list = [single_radius_range,single_radius_range,single_radius_range,single_radius_range]

    # DPC_and_darkField_test.dark_field_with_DPC(
    #     bright_field_angle_range = bright_angle_range, dark_multi_angle_range_list= multi_angle_range_list, dark_multi_radius_range_list= multi_radius_range_list,
    #     bright_n_iter= 1, bright_linear_n_iter= 5, bright_lr= None,
    #     dark_n_iter= 1, dark_linear_n_iter= 15, dark_lr= None,
    #     centre= centre
    # )