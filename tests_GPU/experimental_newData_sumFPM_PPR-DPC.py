## Temporarily adding path
import sys
from pathlib import Path
sys.path.append(str(Path().absolute()))

## === Test Start ===
import os
import numpy as np
import cupy as cp
import math
import json
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.patches as patches
import matplotlib.font_manager as fm

import skimage.io as skio
import tifffile
import csv
import re

from scipy import signal
from pyphaseretrieve.linop  import *
from pyphaseretrieve        import algos
from pyphaseretrieve        import phaseretrieval
from pyphaseretrieve        import loss

FULL_FPM_DAT_FILE_PATH  = '/home/kshen/Ptychography_Project/phase-retrieval-library/phase-retrieval-library/dataset/new_phase_target/0.25NA_1000ms_10X_FPM/fpm.tif'
FPM_0_244_IMAGE         = '/home/kshen/Ptychography_Project/phase-retrieval-library/phase-retrieval-library/dataset/new_phase_target/0.25NA_1000ms_10X_FPM/FPM_centre[520,280]_camerasize256_image[0,244].tif'
FPM_RESOLUTION_0_244_IMAGE_TARGET_1 = '/home/kshen/Ptychography_Project/phase-retrieval-library/phase-retrieval-library/dataset/new_phase_target/0.25NA_1000ms_10X_FPM/FPM_centre[-170,-630]_camerasize256_image[0,244].tif'
FPM_RESOLUTION_0_244_IMAGE_TARGET_2 = '/home/kshen/Ptychography_Project/phase-retrieval-library/phase-retrieval-library/dataset/new_phase_target/0.25NA_1000ms_10X_FPM/FPM_centre[-170,300]_camerasize256_image[0,244].tif'

LED_POS_NA_FILE_PATH    = '/home/kshen/Ptychography_Project/phase-retrieval-library/phase-retrieval-library/dataset/new_phase_target/led_array_pos_na_z65mm.json' ## Local PATH
LED_PATTERNS_FILE_PATH  = '/home/kshen/Ptychography_Project/phase-retrieval-library/phase-retrieval-library/dataset/new_phase_target/led_patterns.json'

X200_CSV_SAVE_PATH = '/home/kshen/Ptychography_Project/phase-retrieval-library/phase-retrieval-library/dataset/new_phase_target/0.25NA_200ms_10X/custom/reconstruction_csv'



## ================================================== Dataset  ====================================================
## ================================================================================================================
class new_custom_dataSet(object):
    def __init__(self, camera_size:int) -> None:
        ## Experiment Setup Parameters Setting (distance and length unit: um)
        # Optical system
        self.wave_lambda        = 0.525
        self.NA                 = 0.25
        # Camera system
        self.camera_size        = camera_size
        self.mag                = 10
        self.camera_pixel_Gsize = 6.5
        self.CAMERA_H_RES       = 2160
        self.CAMERA_V_RES       = 2560
        # LED system
        # self.led_pitch          = 4000
        self.led_d_z            = 67500
        self.led_dia_number     = 19
        self.total_LED_num      = 245
        ## generate experimental setup
        self.experimentSetup()

    # Initialize experimental parameters
    def experimentSetup(self):
        with open(LED_POS_NA_FILE_PATH, 'r') as open_file:
            self.led_pos_na_dict = json.load(open_file)
        with open(LED_PATTERNS_FILE_PATH, 'r') as open_file:
            self.led_patterns_dict = json.load(open_file)

        self.fourier_res           = self.get_fourierResolution()
        self.pupil_mask            = self.get_pupil_mask()
        self.led_sum_na_dict       = self.get_led_total_na_dict()
        self.led_na_dict           = self.get_led_na_dict()
        self.led_r_angle_dict      = self.get_led_r_angle_dict()
        self.total_shifts_pair     = self.get_shiftsPairs()

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
    # ----------------------- Read LED pos and NA -------------------------
    def get_led_total_na_dict(self) -> dict:
        led_sum_na_dict = {}
        for idx, (key, value) in enumerate(self.led_pos_na_dict['led_position_list_na'].items()):
            led_sum_na_dict[key] = np.sqrt(value[0]**2 + value[1]**2)
            if int(key) == self.total_LED_num:
                break
        return led_sum_na_dict
    
    def get_led_na_dict(self) -> dict:
        led_na_dict = {}
        for idx, (key, value) in enumerate(self.led_pos_na_dict['led_position_list_na'].items()):
            led_na_dict[key] = value
            if int(key) == self.total_LED_num:
                break
        return led_na_dict
    
    def get_led_r_angle_dict(self) -> dict:
        led_r_angle_dict = {}
        for idx, (key, value) in enumerate(self.led_pos_na_dict['led_position_list_cartesian'].items()):
            led_r_angle_dict[key] = {}
            led_r_angle_dict[key]['radius'], led_r_angle_dict[key]['angle'] = cart2pol(value[0],value[1])
            if int(key) == self.total_LED_num:
                break
        return led_r_angle_dict
    # ----------------------- NA and reconstruction -----------------------
    def get_reconstruction_size(self, total_selected_led_idx_array:np.ndarray) -> int:
        if total_selected_led_idx_array is not None:
            NA_illu = self.get_illumnationNA_by_selected_index(total_selected_led_idx_array= total_selected_led_idx_array)
        else:
            raise NameError('Incorrect Input')
        synthetic_NA         = NA_illu + self.NA
        reconstruction_size  = math.ceil(2*synthetic_NA/self.wave_lambda/self.fourier_res)

        print('illumination NA: {:.2f}'.format(NA_illu))
        print('Total NA: {:.2f}'.format(synthetic_NA))
        return reconstruction_size 
    
    def get_illumnationNA_by_selected_index(self, total_selected_led_idx_array:np.ndarray):
        temp_na_list = []
        for item in total_selected_led_idx_array:
            temp_na_list.append(self.led_sum_na_dict[str(item)])
            
        illuminationNA  = np.max(temp_na_list)
        return illuminationNA
    # ----------------------- Total shifts and shifts pairs ----------------------- 
    def get_shiftsPairs(self):
        shifts_h_list = []
        shifts_v_list = []
        for led_idx in range(self.total_LED_num):
            shifts_h_list.append(self.led_na_dict[str(led_idx)][0]/ self.wave_lambda / self.fourier_res)
            shifts_v_list.append(-self.led_na_dict[str(led_idx)][1]/ self.wave_lambda / self.fourier_res)
            ##### CHANGE!
        shifts_h = np.array(shifts_h_list)
        shifts_v = np.array(shifts_v_list)
        shifts_pair = np.concatenate([shifts_v.reshape(self.total_LED_num,1),shifts_h.reshape(self.total_LED_num,1)],axis=1)
        return shifts_pair
    # ----------------------- Measurement selection methods ----------------------- 
    def select_multiplex_image_by_multiplex_selectImg_list(self, multiplex_selectImg_list: list):
        print('image loading...')

        img    = skio.imread(FPM_RESOLUTION_0_244_IMAGE_TARGET_2)
        
        img_background = img[0,:,:]
        for idx in range(self.total_LED_num - 1):
            img_background = np.minimum(img_background, img[(idx+1),:,:])

        constant_background = np.mean(img_background)
        print(f'Background value: {constant_background}')

        y = None
        for idx_list, current_image_list in enumerate(multiplex_selectImg_list):
            current_sum_img = np.zeros_like(img[0,:,:])
            for idx_img, image_key in enumerate(current_image_list):
                single_img                                      = img[image_key,:,:] - constant_background
                single_img[single_img<0]                        = 0
                single_img_999_value                            = np.percentile(single_img, 99.9)
                single_img[single_img > single_img_999_value]   = single_img_999_value

                current_sum_img = current_sum_img + single_img
            
            if y is None:
                y = np.uint64(current_sum_img)
            else:
                y = np.concatenate([y,np.uint64(current_sum_img)], axis=0)
        
        print('finish loading...')  
        return y
    
    def get_multiplex_led_array_mask(self, angle_range:list, na_range:np.ndarray, show_angle_map:bool= False) -> np.ndarray:
        multiplex_led_array_mask_list = []
        multiplex_selectImg_list      = []
        for idx, current_angle_range in enumerate(angle_range):
            current_na_range            = na_range[idx]
            current_img_list            = self.single_selectImgList(angle_range= current_angle_range, na_range= current_na_range)
            current_multiplex_mask      = self.single_selectImg2multiplex_array(current_img_list)
            multiplex_selectImg_list.append(current_img_list)
            multiplex_led_array_mask_list.append(current_multiplex_mask)

            if show_angle_map:
                x_pos = []
                y_pos = []
                
                for selected_led_key in current_img_list:
                    x_pos.append(self.led_pos_na_dict['led_position_list_cartesian'][str(selected_led_key)][0])
                    y_pos.append(self.led_pos_na_dict['led_position_list_cartesian'][str(selected_led_key)][1])

                plt.figure()
                plt.scatter(x= x_pos, y= y_pos)
                plt.colorbar()
                plt.title(f'LED pattern {idx+1}')
                plt.xlim(-31,31)
                plt.ylim(-31,31)
                plt.savefig(f'led_pattern/LED pattern {idx+1}')
                plt.close()

        multiplex_led_array_mask                 = np.array(multiplex_led_array_mask_list)
        return multiplex_led_array_mask, multiplex_selectImg_list

    def single_selectImgList(self, angle_range:np.ndarray, na_range:np.ndarray):
        img_list = []
        for key in range(self.total_LED_num):
            current_led_angle       = self.led_r_angle_dict[str(key)]['angle']
            current_led_na          = self.led_sum_na_dict[str(key)]
            if (angle_range[1] - angle_range[0]) < 0:
                if (angle_range[0] < current_led_angle or current_led_angle < angle_range[1]) and (na_range[0] <= current_led_na and current_led_na <= na_range[1]):
                    img_list.append(key)
            else:
                if (angle_range[0] < current_led_angle and current_led_angle < angle_range[1]) and (na_range[0] <= current_led_na and current_led_na <= na_range[1]):
                    img_list.append(key)
        return img_list
        
    def single_selectImg2multiplex_array(self, selectImg_list:list) -> np.ndarray:
        multiplex_array = np.zeros((self.total_LED_num,))
        for idx in range(self.total_LED_num):
            if idx in np.array(selectImg_list):
                multiplex_array[idx] = 1
        return multiplex_array

## ================================================ END Dataset ===================================================
## ================================================================================================================

## ============ functions ============
## ===================================
def cart2pol(x, y):
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)/np.pi * 180
    phi = (phi + 360) % 360
    return(rho, phi)

def delete_file(dir_name):
    dir_name = dir_name
    for file_name in os.listdir(dir_name):
        os.remove(os.path.join(dir_name, file_name))

def read_complex_str_csv(csv_path_file):
    img = None

    with open(csv_path_file, newline='') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',')
        for idx_row, img_row in enumerate(spamreader):
            if img is None:
                img = np.zeros((len(img_row),len(img_row))).astype(np.complex128)
            for idx_col, img_element in enumerate(img_row):
                num_of_element          = re.findall("[-]\d+[.]\d+|\d+[.]\d+", img_element)
                power_of_element        = re.findall("e[+,-]\d+", img_element)
            
                img[idx_row,idx_col]    = float(num_of_element[0])*float('1'+power_of_element[0]) + 1j*(float(num_of_element[1])*float('1'+power_of_element[1]))

    return img
## ============ functions ============
## ===================================

class FPM_summing_PPR_DPC(object):
    def __init__(self,camera_size:int) -> None:
        self.camera_size = camera_size
        self.dataset     = new_custom_dataSet(camera_size= camera_size)

    def PPR_DPC(self, angle_range:np.ndarray, na_range:np.ndarray,
                n_iter= 1, linear_n_iter= 5, lr= 1, centre:list = [0,0], 
                initial_est= None,
                file_header:str= None, for_loop_or_not:bool = False):
        print('DPC start \n----------------------')

        na_range                                           = na_range * self.dataset.NA
        multiplex_led_array_mask, multiplex_selectImg_list = self.dataset.get_multiplex_led_array_mask(angle_range= angle_range, na_range= na_range,show_angle_map= True)
        y                                                  = self.dataset.select_multiplex_image_by_multiplex_selectImg_list(multiplex_selectImg_list= multiplex_selectImg_list)
        y                                                  = cp.array(y)

        for idx in range(int(y.shape[0]/self.camera_size)):
            plt.figure()
            plt.imshow(y[idx*self.camera_size:(idx+1)*self.camera_size,:].get(), cmap=cm.Greys_r)
            plt.colorbar()
            plt.title(f'Sum FPM image {idx}')
            plt.savefig('_recon_img/Sum FPM image {}.png'.format(idx+1))
            plt.close()

        total_selectImg_list = []
        for current_img_list in multiplex_selectImg_list:
            total_selectImg_list = total_selectImg_list + current_img_list

        reconstruction_res          = self.dataset.get_reconstruction_size(total_selected_led_idx_array= total_selectImg_list)
        reconstruction_shape        = (reconstruction_res, reconstruction_res)
        print(f'DPC recontruction shape: {reconstruction_shape}')

        probe                       = cp.array(self.dataset.get_pupil_mask())
        total_shifts_pair           = self.dataset.total_shifts_pair

        if initial_est is None:
            initial_est             = cp.ones(shape= reconstruction_shape, dtype=np.complex128)
        else:
            initial_est             = cp.array(initial_est)
        initial_est             = np.fft.fft2(initial_est, norm="ortho")
        crop_op                 = LinOpCrop2(in_shape= reconstruction_shape, crop_shape= initial_est.shape)
        initial_est             = np.fft.ifftshift(crop_op.applyT(np.fft.fftshift(initial_est)))

        pr_model                = phaseretrieval.MultiplexedPhaseRetrieval(probe= probe,multiplex_led_mask= multiplex_led_array_mask, shifts_pair= total_shifts_pair, reconstruct_shape= reconstruction_shape)
        ppr_method              = algos.PerturbativePhase(pr_model)

        if lr is not None:
            x_est                   = ppr_method.iterate_GradientDescent(y= y, initial_est= initial_est, n_iter= n_iter, linear_n_iter= linear_n_iter, lr=lr)
        else:
            x_est                   = ppr_method.iterate_ConjugateGradientDescent(y= y, initial_est= initial_est, n_iter= n_iter, linear_n_iter= linear_n_iter)

        x_est                   = np.fft.ifft2(x_est, norm="ortho")

        if for_loop_or_not:
            if lr is None:
                file_path   = str(f'_PPR_DPC/CGD/n_iter={n_iter}')
            else:
                file_path   = str(f'_PPR_DPC/GD/n_iter={n_iter}')
        else:
            file_path   = str(f'_recon_img')
        if file_header is None:
            file_header = str('/ResolutionImage_sumFPM_bright_PPR-DPC_')
        na_range = na_range / self.dataset.NA
        if lr is None:
            file_iter   = str(f'n_iter={n_iter}, l_n_iter={linear_n_iter}, NA_range[{na_range[-1][0]}, {na_range[-1][1]}]')
        else:
            file_iter   = str(f'n_iter={n_iter}, l_n_iter={linear_n_iter}, lr={lr}, NA_range[{na_range[-1][0]}, {na_range[-1][1]}]')
        file_end    = str('.png')

        plt.figure()
        plt.imshow(np.abs(x_est.get())**2, cmap=cm.Greys_r)
        plt.colorbar()
        plt.title(f'Intensity: '+ file_iter)
        plt.savefig( file_path + file_header + 'Intensity: ' + file_iter + file_end)
        plt.close()

        plt.figure()
        plt.imshow(np.angle(x_est.get()), cmap=cm.Greys_r)
        plt.colorbar()
        plt.title(f'Phase: '+ file_iter)
        plt.savefig( file_path + file_header + 'Phase: ' + file_iter + file_end)
        plt.close()

        croplinop = LinOpCrop2(in_shape= (self.camera_size*2,self.camera_size*2), crop_shape= reconstruction_shape )
        img = np.abs((np.fft.fftshift(np.fft.fft2(np.angle(x_est.get())))))
        img = np.abs(np.log(img))
        img = croplinop.applyT(img)

        NA_radius   = self.dataset.NA/self.dataset.wave_lambda/self.dataset.fourier_res

        _, axes = plt.subplots()
        plt_circle = plt.Circle((self.camera_size,self.camera_size), NA_radius , fill = False, color='r')
        axes.set_aspect(1)
        axes.add_artist( plt_circle )
        plt.imshow(img, cmap=cm.Greys_r)
        plt.colorbar()
        axes.set_title('Log Abs of FT Phase image '+ file_iter)
        plt.savefig(file_path + file_header + 'Phase: Log FT image ' + file_iter + file_end)
        plt.close()

        phase_img = np.angle(x_est)
        phase_img_99 = np.percentile(np.abs(phase_img),99)
        phase_img[phase_img > phase_img_99]  = 0
        phase_img[phase_img < -phase_img_99] = 0
        plt.figure()
        plt.imshow(phase_img.get(), cmap=cm.Greys_r)
        plt.colorbar()
        plt.title(f'Phase Image, filter 1%: ' + file_iter)
        plt.savefig( file_path + file_header + 'Phase99: ' + file_iter + file_end)
        plt.close()
    
        return x_est      
    
    def auto_CGD_for_loop(self, centre, angle_range, na_range, file_header):
        for n_iter in [1,2,4,6,8,10,15]:
            delete_file(f'_PPR_DPC/CGD/n_iter={n_iter}')
            for linear_n_iter in [1,3,5,10,15,20]:
                _FPM_summing_PPR_DPC.PPR_DPC(angle_range= angle_range, na_range= na_range,
                                 n_iter= n_iter, linear_n_iter= linear_n_iter, centre= centre ,lr= None,
                                 file_header= file_header,for_loop_or_not = True)


                
if __name__ == '__main__':
    ## clean folder
    delete_file('led_pattern')
    delete_file('_recon_img')   

    ## General Setting =============
    _FPM_summing_PPR_DPC = FPM_summing_PPR_DPC(camera_size= 256)

    centre      = [-170,300]
    angle_range = np.array([[0,90],[90,180],[180,270],[270,360]])
    na_range    = np.array([[0,1],[0,1],[0,1],[0,1]])
    # na_range    = np.array([[0,1],[0,1],[0,1],[0,1]])*0.25

    initial_est = _FPM_summing_PPR_DPC.PPR_DPC(angle_range= angle_range, na_range= na_range,
                                 n_iter= 2, linear_n_iter= 3, centre= centre ,lr= None,
                                 file_header = str('/ResolutionImage_sumFPM_bright_PPR-DPC_'),for_loop_or_not = False)


    # centre             = [520,280]
    # angle_range = np.array([[0,90],[90,180],[180,270],[270,360],
    #                         [0,90],[90,180],[180,270],[270,360]])
    # na_range    = np.array([[0,1],[0,1],[0,1],[0,1],
    #                         [1,1.5],[1,1.5],[1,1.5],[1,1.5]])
    # _initial_est = _FPM_summing_PPR_DPC.PPR_DPC(angle_range= angle_range, na_range= na_range,
    #                              n_iter= 2, linear_n_iter= 2, centre= centre ,lr= None,
    #                              initial_est= initial_est,
    #                              file_header = str('/ResolutionImage_sumFPM_dark_PPR-DPC_'),for_loop_or_not = False)
