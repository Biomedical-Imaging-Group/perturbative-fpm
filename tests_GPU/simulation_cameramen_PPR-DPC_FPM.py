## Temporarily adding path
import sys
from pathlib import Path
sys.path.append(str(Path().absolute()))

## === Test Start ===
import os
import numpy as np
# import numpy as cp
import cupy as cp
import math
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.patches as patches
import csv
import re
# from matplotlib_scalebar.scalebar import ScaleBar
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
import matplotlib.font_manager as fm
from scipy import interpolate

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

from pyphaseretrieve.linop  import *
from pyphaseretrieve        import algos
from pyphaseretrieve        import phaseretrieval
from pyphaseretrieve        import loss

DAT_FILE_PATH = '/home/kshen/Ptychography_Project/phase-retrieval-library/phase-retrieval-library/dataset/Simulation' ## Local PATH
FILE_PATH     = '/home/kshen/Ptychography_Project/phase-retrieval-library/phase-retrieval-library/dataset/Simulation/cameraman.tif' ## Local PATH


## ================================================== Dataset  ====================================================
## ================================================================================================================
class U2OS_cell_dataSet(object):
    def __init__(self, camera_size:int) -> None:
        ## Experiment Setup Parameters Setting (distance and length unit: um)
        # Optical system
        self.wave_lambda        = 0.514
        self.NA                 = 0.2
        # Camera system
        self.camera_size        = camera_size
        self.mag                = 8.1485
        self.camera_pixel_Gsize = 6.5
        self.CAMERA_H_RES       = 512
        self.CAMERA_V_RES       = 512
        # LED system
        self.led_pitch          = 4000
        self.led_d_z            = 67500
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
    def get_reconstruction_size(self, bright_field_NA:bool = False, multiplex_led_array_mask:np.ndarray = None, select_led_index_array:np.ndarray= None) -> int:
        if bright_field_NA:
            NA_illu = self.get_bright_illumnationNA()
        elif multiplex_led_array_mask is not None:
            NA_illu = self.get_illumnationNA_by_mulitplex_array(multiplex_led_array_mask= multiplex_led_array_mask)
        elif select_led_index_array is not None:
            NA_illu = self.get_illumnationNA_by_index(select_led_index_array= select_led_index_array)
        else:
             NA_illu = self.get_total_illumnationNA()
        synthetic_NA            = NA_illu + self.NA
        print('illumination NA: {:.2f}'.format(NA_illu))
        print('Total NA: {:.2f}'.format(synthetic_NA))
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
    
    def get_illumnationNA_by_index(self, select_led_index_array:np.ndarray):
        led_index_map       = self.get_led_index_map()
        led_radius_map,_    = self.get_led_r_angle_map()

        radius_list = []
        for idx_item in select_led_index_array:
            index = np.where(led_index_map == idx_item)
            radius_list.append(led_radius_map[index])
        
        led_r           = int(np.max(radius_list)) * self.led_pitch
        illuminationNA  = led_r/math.sqrt(self.led_d_z**2+led_r**2)
        return illuminationNA

    # ----------------------- LED map and mask -----------------------  
    def get_led_bool_mask(self) -> np.ndarray:
        led_ra_size         = math.floor(self.led_dia_number/2)
        led_h_idx_array     = np.linspace(-led_ra_size,led_ra_size,self.led_dia_number)
        led_v_idx_array     = np.linspace(-led_ra_size,led_ra_size,self.led_dia_number)
        
        led_h_idx_map, led_v_idx_map = np.meshgrid(led_h_idx_array,led_v_idx_array)
        led_r_map, _                 = cart2pol(led_h_idx_map,led_v_idx_map)
        led_bool_mask                = led_r_map < self.led_dia_number/2
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
        return led_r_map, led_angle_map
    
        
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
            shifts_pair = np.concatenate([shifts_v.reshape(n_used_led,1), shifts_h.reshape(n_used_led,1)],axis=1)

            return shifts_h_map, shifts_v_map, shifts_pair

    # ----------------------- LEDs selection methods ----------------------- 
    def get_bright_field_LED_map(self):
        sinTheta_h_map, sinTheta_v_map, _ = self.get_shiftsMap_shiftsPairs(return_sinThetaMap_ledMap = True)
        led_NA_map                        = np.sqrt(sinTheta_h_map**2 + sinTheta_v_map**2)

        led_index_map                     = self.get_led_index_map()

        birght_field_led_bool_mask      = (led_NA_map <= self.NA)
        birght_field_led_mask           = birght_field_led_bool_mask.astype(int)
        bright_field_led_index_map      = (birght_field_led_mask * led_index_map).astype(int)
        return bright_field_led_index_map, birght_field_led_mask, birght_field_led_bool_mask
    
    def get_bright_field_multiplex_led_array_mask(self, angle_range:np.ndarray, show_angle_map:bool= False) -> np.ndarray:
        led_r_map, led_angle_map    = self.get_led_r_angle_map()
        led_index_map               = self.get_led_index_map()
        led_bool_mask               = self.get_led_bool_mask()
        _, birght_field_led_mask, _ = self.get_bright_field_LED_map()

        multiplex_led_array_mask_list = []
        for idx in range(angle_range.shape[0]):
            if angle_range[idx,0] is None:
                led_angle_bool_mask = (led_r_map ==0)
            else:
                if (angle_range[idx,1] - angle_range[idx,0])<0:
                    led_angle_bool_mask           = np.logical_and(np.logical_or(angle_range[idx,0] < led_angle_map, led_angle_map < angle_range[idx,1]), led_r_map!=0)
                    # led_angle_bool_mask           = np.logical_and(np.logical_or(angle_range[idx,0] <= led_angle_map, led_angle_map <= angle_range[idx,1]), led_r_map!=0)
                    # led_angle_bool_mask           = np.logical_or(angle_range[idx,0] <= led_angle_map, led_angle_map <= angle_range[idx,1])
                else:
                    led_angle_bool_mask           = np.logical_and(angle_range[idx,0] < led_angle_map, led_angle_map < angle_range[idx,1])
                    # if angle_range[idx,0] == angle_range[idx,1]:
                    #     if angle_range[idx,0] ==0:
                    #         led_angle_bool_mask = np.logical_and( (angle_range[idx,0] == led_angle_map), led_r_map !=0)
                    #     else:
                    #         led_angle_bool_mask = (angle_range[idx,0] == led_angle_map)
                    # led_angle_bool_mask           = np.logical_and(angle_range[idx,0] <= led_angle_map, led_angle_map <= angle_range[idx,1])
                    # led_angle_bool_mask           = np.logical_or( np.logical_and(angle_range[idx,0] <= led_angle_map, led_angle_map <= angle_range[idx,1])  ,  led_r_map==0 )
                    # if angle_range[idx,1] == 360:
                    #     led_angle_bool_mask           = np.logical_or( np.logical_and(angle_range[idx,0] <= led_angle_map, led_angle_map <= angle_range[idx,1])  ,  led_angle_map==0 )
                    # if angle_range[idx,1] == 270:
                    #     led_angle_bool_mask           = np.logical_or( np.logical_and(angle_range[idx,0] <= led_angle_map, led_angle_map <= angle_range[idx,1])  ,  led_r_map==0 )
                    # elif angle_range[idx,1] == 90:
                    #     led_angle_bool_mask           = np.logical_or( np.logical_and(angle_range[idx,0] <= led_angle_map, led_angle_map <= angle_range[idx,1])  ,  led_angle_map==0 )


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
        led_r_map, led_angle_map     = self.get_led_r_angle_map()

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

    def select_FPM_image_by_NA(self, radius:float) -> np.ndarray:
        led_r_map, _    = self.get_led_r_angle_map()
        led_index_map   = self.get_led_index_map()

        select_led_bool_r_map   = (led_r_map<=radius)
        select_led_index_map    = led_index_map*select_led_bool_r_map
        select_led_index_array  = select_led_index_map[select_led_index_map>0].astype(np.int32)

        shifts_pair = self.total_shifts_pair[(select_led_index_array-1),0:2]  

        return select_led_index_array, shifts_pair

    def select_bright_FPM_image(self) -> np.ndarray:
        bright_field_led_index_map,_,_ = self.get_bright_field_LED_map()

        bright_FPM_index_array = bright_field_led_index_map[bright_field_led_index_map>0]
        shifts_pair = self.total_shifts_pair[(bright_FPM_index_array-1),0:2]  
        return bright_FPM_index_array, shifts_pair


## ================================================ END Dataset ===================================================
## ================================================================================================================

## ============ functions ============
## ===================================
def cart2pol(x, y):
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)/np.pi * 180
    phi = (phi + 360) % 360
    return(rho, phi)

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

def P2R(radii, angles):
    return radii * np.exp(1j*angles)

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









## ================================================= START Test Class =================================================
## ====================================================================================================================

class DPC_and_darkField_solver(object):
    def __init__(self,camera_size:int, cpu= True) -> None:
        self.camera_size = camera_size
        self.dataset     = U2OS_cell_dataSet(camera_size= camera_size)
        self.cpu         = cpu

    def select_simulation_range(self, groundtruth_size:int, centre:list = [0,0]):
        SIMULATION_IMAGE_SIZE = 512

        img             = plt.imread(FILE_PATH)
        img             = (img- np.min(img))/(np.max(img) - np.min(img)) - 0.5
        img             = np.exp((1j * img)/5)

        v_center        = SIMULATION_IMAGE_SIZE//2 - centre[1]
        h_center        = SIMULATION_IMAGE_SIZE//2 + centre[0]
        crop_half_size  = groundtruth_size//2

        h_start         = int(h_center - crop_half_size)
        v_start         = int(v_center - crop_half_size)
        crop_size       = groundtruth_size

        groundtruth_img = np.array(img[v_start:v_start+crop_size, h_start:h_start+crop_size])


        # plt.figure()
        # plt.imshow(np.abs(img), cmap=cm.Greys_r)
        # plt.colorbar()
        # plt.title('Simulation target: Abs of Cameraman image')
        # plt.savefig('_recon_img/Simulation_cameraman Abs image.png')
        # plt.close()

        # plt.figure()
        # plt.imshow(np.angle(img), cmap=cm.Greys_r)
        # plt.colorbar()
        # plt.title('Simulation target: Phase of Cameraman image')
        # plt.savefig('_recon_img/Simulation_cameraman Phase image.png')
        # plt.close()

        # plt.figure()
        # plt.imshow(np.abs(groundtruth_img), cmap=cm.Greys_r)
        # plt.colorbar()
        # plt.title('Simulation target: Abs of Cameraman groundtruth image')
        # plt.savefig('_recon_img/Simulation_cameraman Abs groundtruth image.png')
        # plt.close()

        plt.figure()
        plt.imshow(np.angle(groundtruth_img), cmap=cm.Greys_r)
        plt.colorbar()
        plt.title('Simulation target: Phase of Cameraman groundtruth image')
        plt.savefig('_recon_img/Simulation groundtruth Phase image.png')
        plt.close()

        return groundtruth_img


    def Real_DPC(self, bright_field_angle_range:np.ndarray, groundtruth_size:int,  centre:list = [0,0], 
                 alpha= 0,
                 file_pattern= '4 halves'):
        reconstruction_res          = self.dataset.get_reconstruction_size(bright_field_NA= True)
        reconstruction_shape        = (reconstruction_res, reconstruction_res)
        print(f'Real DPC recontruction shape: {reconstruction_shape}')

        simu_img    = self.select_simulation_range(groundtruth_size= groundtruth_size, centre= centre)

        DPC_multiplex_led_array_mask   = self.dataset.get_bright_field_multiplex_led_array_mask(angle_range= bright_field_angle_range, show_angle_map= True)
        total_shifts_pair              = self.dataset.total_shifts_pair
        probe                          = np.array(self.dataset.get_pupil_mask())


        pr_model  = phaseretrieval.MultiplexedPhaseRetrieval(probe= probe, multiplex_led_mask= DPC_multiplex_led_array_mask, shifts_pair= total_shifts_pair, reconstruct_shape= reconstruction_shape)
        DPC_y     = pr_model.apply_ModularSquare(np.fft.fft2(simu_img, norm="ortho"))
        total_image = int(DPC_y.shape[0]/self.camera_size)

        DPC_y_list = []
        for idx_img in range(total_image):
            DPC_y_list.append(DPC_y[idx_img*self.camera_size:(idx_img+1)*self.camera_size,:])
        

        croplinop = LinOpCrop2(in_shape= (reconstruction_shape), crop_shape=(self.camera_size,self.camera_size))
        probe     = np.fft.ifftshift(croplinop.applyT(np.fft.fftshift(probe)))

        transfer_func_list = []
        sum_transfer_func  = np.zeros_like(probe)
        for i_mask in DPC_multiplex_led_array_mask:
            transfer_func = np.zeros_like(probe)
            for idx, mask_item in enumerate(i_mask):
                if (mask_item != False) and (mask_item != 0):
                    pos_shift = LinOpRoll2(total_shifts_pair[idx,0], total_shifts_pair[idx,1])
                    neg_shift = LinOpRoll2(-total_shifts_pair[idx,0], -total_shifts_pair[idx,1])

                    if transfer_func is None:
                        transfer_func = np.copy(neg_shift.apply(probe) - pos_shift.apply(probe))
                    else:
                        transfer_func = transfer_func + neg_shift.apply(probe) - pos_shift.apply(probe)
            sum_transfer_func += transfer_func
            # transfer_func_list.append(transfer_func)

        transfer_func_list.append(sum_transfer_func)
        transfer_func_list.append(sum_transfer_func)


        
        file_path   = str(f'_recon_img/')
        file_header = str('DPC_transfor')
        file_pattern= str('pattern: '+ file_pattern)
        file_end    = str('.png')

        sum_transfer_DPC_phase     = np.zeros_like(probe)
        sum_transfer_func_square   = np.zeros_like(probe)
        for idx, _transfer in enumerate(transfer_func_list):  
            FT_DPC_y  = np.fft.fft2(DPC_y_list[idx], norm= 'ortho')
            FT_DPC_y  = np.fft.ifftshift(croplinop.applyT(np.fft.fftshift(FT_DPC_y)))
            
            sum_transfer_DPC_phase   = sum_transfer_DPC_phase + (_transfer * 1j).conj() * FT_DPC_y
            sum_transfer_func_square = sum_transfer_func_square + np.abs(_transfer)**2
            
            plt.figure()
            plt.imshow(np.fft.fftshift(_transfer), cmap=cm.Greys_r)
            plt.colorbar()
            plt.title(file_header+ f' function{idx}, ' + file_pattern)
            plt.savefig(file_path + file_header + f' function {idx} ' + file_pattern + file_end)
            plt.close()

        alpha       = alpha
        FT_phase    = sum_transfer_DPC_phase/(sum_transfer_func_square + alpha)
        x_est_phase = np.fft.ifft2(FT_phase, norm= 'ortho')/6
        file_alpha  = str(f'alpha={alpha}')

        plt.figure()
        plt.imshow(np.real(x_est_phase), cmap=cm.Greys_r)
        plt.colorbar()
        plt.title(file_header + f' phase, ' + file_pattern)
        plt.savefig(file_path + file_header + ' phase, ' + file_alpha + f', '+ file_pattern + file_end)
        plt.close()


        croplinop                = LinOpCrop2((groundtruth_size,groundtruth_size), reconstruction_shape)
        down_sample_GT           = croplinop.apply(np.fft.fftshift(np.fft.fft2(simu_img, norm='ortho')))
        down_sample_GT_phase     = np.angle(np.fft.ifft2(np.fft.ifftshift(down_sample_GT), norm='ortho'))
        crop_phase_FT_ground     = np.abs(np.fft.fftshift(np.fft.fft2(down_sample_GT_phase, norm='ortho')))

        phase_FT_xest            = np.abs((np.fft.fftshift(np.fft.fft2(np.real(x_est_phase), norm="ortho"))))
        error_map                = np.abs((phase_FT_xest - crop_phase_FT_ground)/(crop_phase_FT_ground))
        error_map[error_map > 1] = 1
        error_map                = np.pad(error_map,((int(np.floor((groundtruth_size-reconstruction_shape[0])/2)), int(np.ceil((groundtruth_size-reconstruction_shape[0])/2))), (0, 0)),mode='constant', constant_values=1)
        error_map                = np.pad(error_map,((0, 0), (int(np.floor((groundtruth_size-reconstruction_shape[0])/2)), int(np.ceil((groundtruth_size-reconstruction_shape[0])/2)))),mode='constant', constant_values=1)

        phase_FT_ground         = np.abs(np.fft.fftshift(np.fft.fft2(np.angle(simu_img), norm= 'ortho')))
        expand_phase_FT_xest    = croplinop.applyT(phase_FT_xest)
        noise_map               = expand_phase_FT_xest - phase_FT_ground
        snr_value               = 10*np.log10(np.mean(phase_FT_ground**2)/np.mean((noise_map)**2))
        plt.figure()
        plt.imshow(error_map, cmap=cm.Greys_r)
        plt.colorbar()
        plt.title('FT Error map, SNR: {:.2f}'.format(snr_value))
        plt.savefig(file_path + file_header + ' Phase: Error map ' + file_pattern + '_' + file_alpha + file_end)
        plt.close()



    def DPC(self, bright_field_angle_range:np.ndarray, groundtruth_size:int, n_iter= 1, linear_n_iter= 1, lr= 5, centre:list = [0,0], for_loop_or_not:bool = False, alpha=0, _lambda=0):
        print('DPC start \n----------------------')

        reconstruction_res          = self.dataset.get_reconstruction_size(bright_field_NA= True)
        reconstruction_shape        = (reconstruction_res, reconstruction_res)
        print(f'DPC recontruction shape: {reconstruction_shape}')

        simu_img    = self.select_simulation_range(groundtruth_size= groundtruth_size, centre= centre)

        self.DPC_multiplex_led_array_mask   = self.dataset.get_bright_field_multiplex_led_array_mask(angle_range= bright_field_angle_range, show_angle_map= True)
        probe                               = cp.array(self.dataset.get_pupil_mask())
        total_shifts_pair                   = self.dataset.total_shifts_pair

        pr_model                            = phaseretrieval.MultiplexedPhaseRetrieval(probe= probe, multiplex_led_mask= self.DPC_multiplex_led_array_mask, shifts_pair= total_shifts_pair, reconstruct_shape= reconstruction_shape)

        DPC_y                               = pr_model.apply_ModularSquare(np.fft.fft2(cp.array(simu_img), norm="ortho"))

        for idx in range(int(DPC_y.shape[0]/self.camera_size)):
            plt.figure()
            plt.imshow(DPC_y[idx*self.camera_size:(idx+1)*self.camera_size,:].get(), cmap=cm.Greys_r)
            plt.colorbar()
            plt.title('Generated output: intensity measurement')
            plt.savefig('_recon_img/Simulation_Generated measurement {}.png'.format(idx+1))
            plt.close()

        initial_est             = cp.ones(shape= reconstruction_shape, dtype=np.complex128)
        initial_est             = np.fft.fft2(initial_est, norm="ortho")

        ppr_method              = algos.PerturbativePhase(pr_model)

        if lr is not None:
            x_est                   = ppr_method.iterate_GradientDescent(y= DPC_y, initial_est= initial_est, n_iter= n_iter, linear_n_iter= linear_n_iter, lr=lr, alpha= alpha)
        else:
            x_est                   = ppr_method.iterate_ConjugateGradientDescent(y= DPC_y, initial_est= initial_est, n_iter= n_iter, linear_n_iter= linear_n_iter, tolerance= 1e-20, _lambda=_lambda)

        x_est                   = np.fft.ifft2(x_est, norm="ortho")

        if self.cpu:
            x_est       = cp.asnumpy(x_est)


        if for_loop_or_not:
            if lr is None:
                file_path   = str(f'_DPC/CGD/n_iter={n_iter}')
            else:
                file_path   = str(f'_DPC/GD/n_iter={n_iter}')
        else:
            file_path   = str(f'_recon_img')
        file_header = str('/Simulation_DPC_4Q_')
        if lr is None:
            file_iter   = str(f'n_iter={n_iter}, l_n_iter={linear_n_iter}, NA_range[0,1]')
        else:
            file_iter   = str(f'n_iter={n_iter}, l_n_iter={linear_n_iter}, lr={lr}, NA_range[0,1]')
        file_end    = str('.png')


        plt.figure()
        plt.imshow(np.abs(x_est)**2, cmap=cm.Greys_r)
        plt.colorbar()
        plt.title(f'Intensity: '+ file_iter)
        plt.savefig( file_path + file_header + 'Intensity: ' + file_iter + file_end)
        plt.close()

        plt.figure()
        plt.imshow(np.angle(x_est), cmap=cm.Greys_r)
        plt.colorbar()
        plt.title(f'Phase: '+ file_iter)
        plt.savefig( file_path + file_header + 'Phase: ' + file_iter + file_end)
        plt.close()

        croplinop           = LinOpCrop2((groundtruth_size,groundtruth_size), reconstruction_shape)
        phase_FT_xest       = np.abs((np.fft.fftshift(np.fft.fft2(np.angle(x_est), norm="ortho"))))
        log_FT_phase_xest   = np.log(phase_FT_xest)
        log_FT_phase_xest   = croplinop.applyT(log_FT_phase_xest)
        plt.figure()
        plt.imshow(log_FT_phase_xest, cmap=cm.Greys_r)
        plt.colorbar()
        plt.title('Abs Phase Log FT image')
        plt.savefig(file_path + file_header + 'Phase: Log FT image ' + file_iter + file_end)
        plt.close()

        down_sample_GT       = croplinop.apply(np.fft.fftshift(np.fft.fft2(simu_img, norm='ortho')))
        down_sample_GT_phase = np.angle(np.fft.ifft2(np.fft.ifftshift(down_sample_GT), norm='ortho'))
        crop_phase_FT_ground = np.abs(np.fft.fftshift(np.fft.fft2(down_sample_GT_phase, norm='ortho')))

        error_map = np.abs((phase_FT_xest - crop_phase_FT_ground)/(crop_phase_FT_ground))
        error_map[error_map > 1] = 1
        error_map = np.pad(error_map,((int(np.floor((groundtruth_size-reconstruction_shape[0])/2)), int(np.ceil((groundtruth_size-reconstruction_shape[0])/2))), (0, 0)),mode='constant', constant_values=1)
        error_map = np.pad(error_map,((0, 0), (int(np.floor((groundtruth_size-reconstruction_shape[0])/2)), int(np.ceil((groundtruth_size-reconstruction_shape[0])/2)))),mode='constant', constant_values=1)

        phase_FT_ground = np.abs(np.fft.fftshift(np.fft.fft2(np.angle(simu_img), norm= 'ortho')))
        expand_phase_FT_xest = croplinop.applyT(phase_FT_xest)
        noise_map = expand_phase_FT_xest - phase_FT_ground
        snr_value = 10*np.log10(np.mean(phase_FT_ground**2)/np.mean((noise_map)**2))
        plt.figure()
        plt.imshow(error_map, cmap=cm.Greys_r)
        plt.colorbar()
        plt.title('FT Error map, SNR: {:.2f}'.format(snr_value))
        plt.savefig(file_path + file_header + 'Phase: Error map ' + file_iter + file_end)
        plt.close()

        return x_est

    def dark_field_with_DPC(
            self, bright_field_angle_range:np.ndarray, dark_multi_angle_range_list:list, dark_multi_radius_range_list:list, groundtruth_size:int,
            bright_n_iter= 1, bright_linear_n_iter= 5, bright_lr= None,
            dark_n_iter= 15, dark_linear_n_iter= 5, dark_lr= None,
            centre:list= [0,0], read_exsiting_csv:str= None, initial_est= None, ones_initial_est:bool= False, NA_range=[1,1],
            for_loop_or_not:bool = False):

        self.DPC_multiplex_led_array_mask     = self.dataset.get_bright_field_multiplex_led_array_mask(angle_range= bright_field_angle_range, show_angle_map= True)
        self.dark_multiplex_led_array_mask    = self.dataset.get_dark_multiplex_led_array_mask(
            multi_angle_range= dark_multi_angle_range_list, multi_radius_range= dark_multi_radius_range_list, show_angle_map= True)

        probe                       = cp.array(self.dataset.get_pupil_mask())
        total_shifts_pair           = self.dataset.total_shifts_pair
        
        total_multiplex_led_mask    = np.concatenate((self.DPC_multiplex_led_array_mask, self.dark_multiplex_led_array_mask), axis=0)

        reconstruction_size                = self.dataset.get_reconstruction_size(multiplex_led_array_mask= self.dark_multiplex_led_array_mask)
        reconstruction_shape               = (reconstruction_size, reconstruction_size)
        print(f'Dark recontruction shape: {reconstruction_shape}')

        if ones_initial_est:
            initial_est         = cp.ones(shape= reconstruction_shape, dtype=np.complex128)
        else:
            if initial_est is None:
                if read_exsiting_csv is not None:
                    initial_est = read_complex_str_csv('/home/kshen/Ptychography_Project/phase-retrieval-library/phase-retrieval-library/dataset/Simulation/'+read_exsiting_csv)
                    initial_est = cp.array(initial_est)
                else:
                    initial_est = self.DPC(bright_field_angle_range= bright_field_angle_range, groundtruth_size= groundtruth_size, n_iter= bright_n_iter, linear_n_iter= bright_linear_n_iter, lr= bright_lr, centre= centre)

        initial_est = np.fft.fft2(initial_est, norm="ortho")
        print(f'Initial guess shape: {initial_est.shape}')
        crop_op     = LinOpCrop2(in_shape= reconstruction_shape, crop_shape= initial_est.shape)
        initial_est = np.fft.ifftshift(crop_op.applyT(np.fft.fftshift(initial_est)))

        pr_model                    = phaseretrieval.MultiplexedPhaseRetrieval(probe= probe,multiplex_led_mask= total_multiplex_led_mask, shifts_pair= total_shifts_pair, reconstruct_shape= reconstruction_shape)
        
        simu_img    = self.select_simulation_range(groundtruth_size= groundtruth_size, centre= centre)
        simu_img    = cp.array(simu_img)
        total_y     = pr_model.apply_ModularSquare(np.fft.fft2(simu_img, norm="ortho"))

        for idx in range(int(total_y.shape[0]/self.camera_size)):
            plt.figure()
            plt.imshow(total_y[idx*self.camera_size:(idx+1)*self.camera_size,:].get(), cmap=cm.Greys_r)
            plt.colorbar()
            plt.title('Generated output: intensity measurement')
            plt.savefig('_recon_img/Simulation_Generated measurement {}.png'.format(idx+1))
            plt.close()


        ppr_method                  = algos.PerturbativePhase(pr_model)
        if dark_lr is not None:
            x_est               = ppr_method.iterate_GradientDescent(y= total_y, initial_est= initial_est, n_iter= dark_n_iter, linear_n_iter= dark_linear_n_iter, lr= dark_lr)
        else:
            x_est               = ppr_method.iterate_ConjugateGradientDescent(y= total_y, initial_est= initial_est, n_iter= dark_n_iter, linear_n_iter= dark_linear_n_iter)

        x_est                   = np.fft.ifft2(x_est, norm="ortho")

        if for_loop_or_not:
            if dark_lr is None:
                file_path   = str(f'_PPR_DPC/CGD/n_iter={dark_n_iter}')
            else:
                file_path   = str(f'_PPR_DPC/GD/n_iter={dark_n_iter}')
        else:
            file_path   = str(f'_recon_img')
        file_header = str('/Simulation_Dark_PPR-DPC_')
        if dark_lr is None:
            file_iter   = str(f'n_iter={dark_n_iter}, l_n_iter={dark_linear_n_iter}, NA_range[{NA_range[0]}, {NA_range[1]}]')
        else:
            file_iter   = str(f'n_iter={dark_n_iter}, l_n_iter={dark_linear_n_iter}, lr={dark_lr}, NA_range[{NA_range[0]}, {NA_range[1]}]')
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

        croplinop           = LinOpCrop2((groundtruth_size,groundtruth_size), reconstruction_shape)
        phase_FT_xest       = np.abs((np.fft.fftshift(np.fft.fft2(np.angle(x_est), norm="ortho"))))
        log_FT_phase_xest   = np.log(phase_FT_xest)
        log_FT_phase_xest   = croplinop.applyT(log_FT_phase_xest)
        plt.figure()
        plt.imshow(log_FT_phase_xest.get(), cmap=cm.Greys_r)
        plt.colorbar()
        plt.title('Abs Phase Log FT image')
        plt.savefig(file_path + file_header + 'Phase: Log FT image ' + file_iter + file_end)
        plt.close()

        down_sample_GT       = croplinop.apply(np.fft.fftshift(np.fft.fft2(simu_img, norm='ortho')))
        down_sample_GT_phase = np.angle(np.fft.ifft2(np.fft.ifftshift(down_sample_GT), norm='ortho'))
        crop_phase_FT_ground = np.abs(np.fft.fftshift(np.fft.fft2(down_sample_GT_phase, norm='ortho')))

        error_map = np.abs((phase_FT_xest - crop_phase_FT_ground)/(crop_phase_FT_ground))
        error_map[error_map > 1] = 1
        error_map = np.pad(error_map,((int(np.floor((groundtruth_size-reconstruction_shape[0])/2)), int(np.ceil((groundtruth_size-reconstruction_shape[0])/2))), (0, 0)),mode='constant', constant_values=1)
        error_map = np.pad(error_map,((0, 0), (int(np.floor((groundtruth_size-reconstruction_shape[0])/2)), int(np.ceil((groundtruth_size-reconstruction_shape[0])/2)))),mode='constant', constant_values=1)

        phase_FT_ground = np.abs(np.fft.fftshift(np.fft.fft2(np.angle(simu_img), norm= 'ortho')))
        expand_phase_FT_xest = croplinop.applyT(phase_FT_xest)
        noise_map = expand_phase_FT_xest - phase_FT_ground
        snr_value = 10*np.log10(np.mean(phase_FT_ground**2)/np.mean((noise_map)**2))
        plt.figure()
        plt.imshow(error_map.get(), cmap=cm.Greys_r)
        plt.colorbar()
        plt.title('FT Error map, SNR: {:.2f}'.format(snr_value))
        plt.savefig(file_path + file_header + 'Phase: Error map ' + file_iter + file_end)
        plt.close()

        return x_est
    



    ### ================================================================================================================================================
    ### ================================================================================================================================================
    ### ======================================================= DPC and PPR-DPC ----------------- FPM ==================================================
    ### ================================================================================================================================================
    ### ================================================================================================================================================


    
    def bright_FPM(self, groundtruth_size, n_iter= 5, lr= 1, centre=[0,0], amp_based_or_not:bool = True, for_loop_or_not:bool = False):
        print('FPM Start \n----------------------')

        simu_img                    = self.select_simulation_range(groundtruth_size= groundtruth_size, centre= centre)
        simu_img                    = cp.array(simu_img)

        img_idx_array, shifts_pair  = self.dataset.select_bright_FPM_image()
        probe                       = cp.array(self.dataset.get_pupil_mask())

        reconstruction_res          = self.dataset.get_reconstruction_size(bright_field_NA= True)
        reconstruction_shape        = (reconstruction_res,reconstruction_res)
        print(f'recontruction size: {reconstruction_shape}')

        pr_model                    = phaseretrieval.FourierPtychography2d(probe= probe, shifts_pair= shifts_pair, reconstruct_shape= reconstruction_shape)
        y                           = pr_model.apply_ModularSquare(np.fft.fft2(simu_img, norm="ortho"))
        print(y.shape)

        initial_est                 = cp.ones(shape= reconstruction_shape, dtype=np.complex128)
        initial_est                 = np.fft.fft2(initial_est, norm="ortho")

        if amp_based_or_not:
            loss_function               = loss.loss_amplitude_based(epsilon= 1e-4)
            gd_method                   = algos.GradientDescent(pr_model, loss_func= loss_function, line_search= None)
        else:
            gd_method                   = algos.GradientDescent(pr_model, loss_func= None, line_search= True)
        x_est                       = gd_method.iterate(y = y, initial_est = initial_est, n_iter = n_iter, lr = lr)
        x_est                       = np.fft.ifft2(x_est, norm="ortho")


        if for_loop_or_not:
            if amp_based_or_not:
                file_path   = str(f'_FPM/amp_based/n_iter={n_iter}')
            else:
                file_path   = str(f'_FPM/intensity_based/n_iter={n_iter}')
        else:
            file_path   = str(f'_recon_img')
        file_header = str('/Simulation_Bright_FPM_')
        if amp_based_or_not:
            file_iter   = str(f'ampBased_n_iter={n_iter}_lr={lr}')
        else:
            file_iter   = str(f'intenBased_n_iter={n_iter}_lr=Auto')
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

        croplinop           = LinOpCrop2((groundtruth_size,groundtruth_size), reconstruction_shape)
        phase_FT_xest       = np.abs((np.fft.fftshift(np.fft.fft2(np.angle(x_est), norm="ortho"))))
        log_FT_phase_xest   = np.log(phase_FT_xest)
        log_FT_phase_xest   = croplinop.applyT(log_FT_phase_xest)
        plt.figure()
        plt.imshow(log_FT_phase_xest.get(), cmap=cm.Greys_r)
        plt.colorbar()
        plt.title('Abs Phase Log FT image')
        plt.savefig(file_path + file_header + 'Phase: Log FT image ' + file_iter + file_end)
        plt.close()

        down_sample_GT       = croplinop.apply(np.fft.fftshift(np.fft.fft2(simu_img, norm='ortho')))
        down_sample_GT_phase = np.angle(np.fft.ifft2(np.fft.ifftshift(down_sample_GT), norm='ortho'))
        crop_phase_FT_ground = np.abs(np.fft.fftshift(np.fft.fft2(down_sample_GT_phase, norm='ortho')))

        error_map = np.abs((phase_FT_xest - crop_phase_FT_ground)/(crop_phase_FT_ground))
        error_map[error_map > 1] = 1
        error_map = np.pad(error_map,((int(np.floor((groundtruth_size-reconstruction_shape[0])/2)), int(np.ceil((groundtruth_size-reconstruction_shape[0])/2))), (0, 0)),mode='constant', constant_values=1)
        error_map = np.pad(error_map,((0, 0), (int(np.floor((groundtruth_size-reconstruction_shape[0])/2)), int(np.ceil((groundtruth_size-reconstruction_shape[0])/2)))),mode='constant', constant_values=1)

        phase_FT_ground = np.abs(np.fft.fftshift(np.fft.fft2(np.angle(simu_img), norm= 'ortho')))
        expand_phase_FT_xest = croplinop.applyT(phase_FT_xest)
        noise_map = expand_phase_FT_xest - phase_FT_ground
        snr_value = 10*np.log10(np.mean(phase_FT_ground**2)/np.mean((noise_map)**2))
        plt.figure()
        plt.imshow(error_map.get(), cmap=cm.Greys_r)
        plt.colorbar()
        plt.title('FT Error map, SNR: {:.2f}'.format(snr_value))
        plt.savefig(file_path + file_header + 'Phase: Error map ' + file_iter + file_end)
        plt.close()


    def FPM(self, groundtruth_size, radius:float, n_iter= 5, lr= 1, centre=[0,0],amp_based_or_not:bool = True, NA_range:list= None,for_loop_or_not:bool = False):
        print('FPM Start \n----------------------')

        simu_img                    = self.select_simulation_range(groundtruth_size= groundtruth_size, centre= centre)
        simu_img                    = cp.array(simu_img)

        img_idx_array, shifts_pair  = self.dataset.select_FPM_image_by_NA(radius= radius)
        probe                       = cp.array(self.dataset.get_pupil_mask())

        reconstruction_res          = self.dataset.get_reconstruction_size(select_led_index_array= img_idx_array)
        reconstruction_shape        = (reconstruction_res,reconstruction_res)
        print(f'recontruction size: {reconstruction_shape}')

        pr_model                    = phaseretrieval.FourierPtychography2d(probe= probe, shifts_pair= shifts_pair, reconstruct_shape= reconstruction_shape)
        y                           = pr_model.apply_ModularSquare(np.fft.fft2(simu_img, norm="ortho"))
        print(y.shape)

        initial_est                 = cp.ones(shape= reconstruction_shape, dtype=np.complex128)
        initial_est                 = np.fft.fft2(initial_est, norm="ortho")

        if amp_based_or_not:
            loss_function               = loss.loss_amplitude_based(epsilon= 1e-4)
            gd_method                   = algos.GradientDescent(pr_model, loss_func= loss_function, line_search= None)
        else:
            gd_method                   = algos.GradientDescent(pr_model, loss_func= None, line_search= True)
        x_est                       = gd_method.iterate(y = y, initial_est = initial_est, n_iter = n_iter, lr = lr)
        x_est                       = np.fft.ifft2(x_est, norm="ortho")

        if for_loop_or_not:
            if amp_based_or_not:
                file_path   = str(f'_FPM/amp_based/n_iter={n_iter}')
            else:
                file_path   = str(f'_FPM/intensity_based/n_iter={n_iter}')
        else:
            file_path   = str(f'_recon_img')
        file_header = str('/Simulation_FPM_')
        if amp_based_or_not:
            file_iter   = str(f'ampBased_n_iter={n_iter}_lr={lr}')
        else:
            file_iter   = str(f'intenBased_n_iter={n_iter}_lr=Auto')
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

        croplinop           = LinOpCrop2((groundtruth_size,groundtruth_size), reconstruction_shape)
        phase_FT_xest       = np.abs((np.fft.fftshift(np.fft.fft2(np.angle(x_est), norm="ortho"))))
        log_FT_phase_xest   = np.log(phase_FT_xest)
        log_FT_phase_xest   = croplinop.applyT(log_FT_phase_xest)
        plt.figure()
        plt.imshow(log_FT_phase_xest.get(), cmap=cm.Greys_r)
        plt.colorbar()
        plt.title('Abs Phase Log FT image')
        plt.savefig(file_path + file_header + 'Phase: Log FT image ' + file_iter + file_end)
        plt.close()

        down_sample_GT       = croplinop.apply(np.fft.fftshift(np.fft.fft2(simu_img, norm='ortho')))
        down_sample_GT_phase = np.angle(np.fft.ifft2(np.fft.ifftshift(down_sample_GT), norm='ortho'))
        crop_phase_FT_ground = np.abs(np.fft.fftshift(np.fft.fft2(down_sample_GT_phase, norm='ortho')))

        error_map = np.abs((phase_FT_xest - crop_phase_FT_ground)/(crop_phase_FT_ground))
        error_map[error_map > 1] = 1
        error_map = np.pad(error_map,((int(np.floor((groundtruth_size-reconstruction_shape[0])/2)), int(np.ceil((groundtruth_size-reconstruction_shape[0])/2))), (0, 0)),mode='constant', constant_values=1)
        error_map = np.pad(error_map,((0, 0), (int(np.floor((groundtruth_size-reconstruction_shape[0])/2)), int(np.ceil((groundtruth_size-reconstruction_shape[0])/2)))),mode='constant', constant_values=1)

        phase_FT_ground = np.abs(np.fft.fftshift(np.fft.fft2(np.angle(simu_img), norm= 'ortho')))
        expand_phase_FT_xest = croplinop.applyT(phase_FT_xest)
        noise_map = expand_phase_FT_xest - phase_FT_ground
        snr_value = 10*np.log10(np.mean(phase_FT_ground**2)/np.mean((noise_map)**2))
        plt.figure()
        plt.imshow(error_map.get(), cmap=cm.Greys_r)
        plt.colorbar()
        plt.title('FT Error map, SNR: {:.2f}'.format(snr_value))
        plt.savefig(file_path + file_header + 'Phase: Error map ' + file_iter + file_end)
        plt.close()

    
## ================================================== END Test Class ==================================================
## ====================================================================================================================

if __name__ == '__main__':
    ## clean folder
    delete_file('led_pattern')
    delete_file('_recon_img')   

    _simulation = DPC_and_darkField_solver(camera_size= 100)


    centre              = [50,60]
    groundtruth_size    = 256
    ## 1. DPC ==========================
    ## =================================
    # bright_angle_range = np.array([[0,180],[90,270]])
    # bright_angle_range = np.array([[0,180],[270,90]])
    # pattern = '2 halves 90 degree'
    # for alpha in np.geomspace(1e+2, 1e-3, num=6):
    #     _simulation.Real_DPC(bright_field_angle_range= bright_angle_range, groundtruth_size= groundtruth_size, centre= centre, 
    #                          alpha= alpha, file_pattern = pattern)
        
    # bright_angle_range = np.array([[0,90],[90,180]])
    # pattern = '2 quarters 90 degree'
    # for alpha in np.geomspace(1e+2, 1e-3, num=6):
    #     _simulation.Real_DPC(bright_field_angle_range= bright_angle_range, groundtruth_size= groundtruth_size, centre= centre, 
    #                          alpha= alpha, file_pattern = pattern)

    # _simulation.Real_DPC(bright_field_angle_range= bright_angle_range, groundtruth_size= groundtruth_size, centre= centre, alpha= 1e-3)


    ## 2. bright PPR-DPC ==========================
    ## ============================================
    # bright_angle_range = np.array([[0,90],[90,180],[180,270],[270,360],[0,0],[90,90],[180,180],[270,270],[None,None]])
    # bright_angle_range = np.array([[0,90],[90,180],[180,270],[270,360]])
    # bright_angle_range = np.array([[0,180],[180,360],[90,270],[270,90]])
    # bright_angle_range = np.array([[0,90],[90,180]])
    bright_angle_range = np.array([[0,180],[90,270]])
    # bright_angle_range = np.array([[0,90]])

    _simulation.DPC(bright_field_angle_range= bright_angle_range, groundtruth_size= groundtruth_size, 
                    n_iter= 1, linear_n_iter= 100, centre= centre, lr= None, _lambda= 0)

    # for n_iter in [1,25,50,100]:
        # delete_file(f'_DPC/CGD/n_iter={n_iter}')
    #     for linear_n_iter in [100]:
    #         _simulation.DPC(bright_field_angle_range= bright_angle_range, groundtruth_size= groundtruth_size, 
    #         n_iter= n_iter, linear_n_iter= linear_n_iter, centre= centre, lr= None, for_loop_or_not= True)

    # n_iter = 25
    # linear_n_iter = 25
    # delete_file(f'_DPC/GD/n_iter={n_iter}')
    # for lr in [1.6e-4]:
    #     _simulation.DPC(bright_field_angle_range= bright_angle_range, groundtruth_size= groundtruth_size, 
    #     n_iter= n_iter, linear_n_iter= linear_n_iter, centre= centre, lr= lr, for_loop_or_not= True)

    # _simulation.DPC(bright_field_angle_range= bright_angle_range, groundtruth_size= groundtruth_size, 
    #                 n_iter= 1, linear_n_iter= 1000, centre= centre, lr= 1.6e-4, alpha= 0.1)

    ## 2. PDPC =========================
    ## =================================


    # centre             = [50,60]
    # bright_angle_range = np.array([[0,90],[90,180],[180,270],[270,360]])
    # groundtruth_size     = 256

    # multi_angle_range_list  = [[0,90],[90,180],[180,270],[270,360]]

    # outer_NA_facter = 1.5
    # inner_NA = _simulation.dataset.NA * 0
    # outer_NA = _simulation.dataset.NA * outer_NA_facter
    # inner_dark_radius       = (inner_NA*_simulation.dataset.led_d_z)/np.sqrt(1-inner_NA**2)/_simulation.dataset.led_pitch
    # outer_dark_radius       = (outer_NA*_simulation.dataset.led_d_z)/np.sqrt(1-outer_NA**2)/_simulation.dataset.led_pitch
    # single_radius_range     = [np.sqrt(2*(inner_dark_radius**2)), np.sqrt(2*(outer_dark_radius**2))]
    # multi_radius_range_list = [single_radius_range,single_radius_range,single_radius_range,single_radius_range]


    # x_est = _simulation.dark_field_with_DPC(
    #     bright_field_angle_range= bright_angle_range, 
    #     dark_multi_angle_range_list= multi_angle_range_list, dark_multi_radius_range_list= multi_radius_range_list, 
    #     groundtruth_size= groundtruth_size,
    #     bright_n_iter= 100, bright_linear_n_iter= 100, bright_lr= None,
    #     dark_n_iter= 1, dark_linear_n_iter= 100, dark_lr= None,
    #     centre= centre, read_exsiting_csv= '4_quarter_bright_PPR_DPC_117x117_csv_file.csv', NA_range= [1, outer_NA_facter]
    # )

    # for outer_NA_facter in [2]:
        
    #     multi_angle_range_list  = [[0,90],[90,180],[180,270],[270,360]]

    #     inner_NA = _simulation.dataset.NA * 0
    #     outer_NA = _simulation.dataset.NA * outer_NA_facter
    #     inner_dark_radius       = (inner_NA*_simulation.dataset.led_d_z)/np.sqrt(1-inner_NA**2)/_simulation.dataset.led_pitch
    #     outer_dark_radius       = (outer_NA*_simulation.dataset.led_d_z)/np.sqrt(1-outer_NA**2)/_simulation.dataset.led_pitch
    #     single_radius_range     = [np.sqrt(2*(inner_dark_radius**2)), np.sqrt(2*(outer_dark_radius**2))]
    #     multi_radius_range_list = [single_radius_range,single_radius_range,single_radius_range,single_radius_range]

    #     x_est = _simulation.dark_field_with_DPC(
    #         bright_field_angle_range = bright_angle_range, dark_multi_angle_range_list= multi_angle_range_list, dark_multi_radius_range_list= multi_radius_range_list, groundtruth_size= groundtruth_size,
    #         bright_n_iter= 100, bright_linear_n_iter= 100, bright_lr= None,
    #         dark_n_iter= 500, dark_linear_n_iter= 100, dark_lr= None,
    #         centre= centre, read_exsiting_csv= '4_quarter_bright_PPR_DPC_117x117_csv_file.csv', NA_range= [1, outer_NA_facter]
    #     )

    #     dark_multiplex_led_array_mask    = _simulation.dataset.get_dark_multiplex_led_array_mask(
    #         multi_angle_range= multi_angle_range_list, multi_radius_range= multi_radius_range_list, show_angle_map= False)
    #     reconstruction_size = _simulation.dataset.get_reconstruction_size(multiplex_led_array_mask= dark_multiplex_led_array_mask)
    #     np.savetxt('/home/kshen/Ptychography_Project/phase-retrieval-library/phase-retrieval-library/dataset/Simulation/4_quarter_dark_DPC_'
    #                    + 'NA_range[{}, {}]_'.format(0,outer_NA_facter)+ str(reconstruction_size) +'x'+ str(reconstruction_size) +'_csv_file.csv',x_est.get(), delimiter=',')


    # multi_angle_range_list  = [[0,180],[180,360],[90,270],[270,90]]
    # inner_NA = _simulation.dataset.NA * 1.15
    # outer_NA = _simulation.dataset.NA * 1.3
    # inner_dark_radius       = (inner_NA*_simulation.dataset.led_d_z)/np.sqrt(1-inner_NA**2)/_simulation.dataset.led_pitch
    # outer_dark_radius       = (outer_NA*_simulation.dataset.led_d_z)/np.sqrt(1-outer_NA**2)/_simulation.dataset.led_pitch
    # single_radius_range     = [np.sqrt(2*(inner_dark_radius**2)), np.sqrt(2*(outer_dark_radius**2))]
    # multi_radius_range_list = [single_radius_range,single_radius_range,single_radius_range,single_radius_range]

    # dark_multiplex_led_array_mask    = _simulation.dataset.get_dark_multiplex_led_array_mask(
    #         multi_angle_range= multi_angle_range_list, multi_radius_range= multi_radius_range_list, show_angle_map= True)
   

    # # ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # second_multi_angle_range_list  = [[0,90],[90,180],[180,270],[270,360],
    #                                   [0,90],[90,180],[180,270],[270,360]]

    # second_inner_NA = _simulation.dataset.NA * 1.3
    # second_outer_NA = _simulation.dataset.NA * 1.5
    # second_inner_dark_radius       = (second_inner_NA*_simulation.dataset.led_d_z)/np.sqrt(1-second_inner_NA**2)/_simulation.dataset.led_pitch
    # second_outer_dark_radius       = (second_outer_NA*_simulation.dataset.led_d_z)/np.sqrt(1-second_outer_NA**2)/_simulation.dataset.led_pitch
    # second_single_radius_range     = [np.sqrt(2*(second_inner_dark_radius**2)), np.sqrt(2*(second_outer_dark_radius**2))]
    # second_multi_radius_range_list = [single_radius_range,single_radius_range,single_radius_range,single_radius_range,
    #                                   second_single_radius_range,second_single_radius_range,second_single_radius_range,second_single_radius_range]

    # x_est = _simulation.dark_field_with_DPC(
    #     bright_field_angle_range = bright_angle_range, dark_multi_angle_range_list= second_multi_angle_range_list, dark_multi_radius_range_list= second_multi_radius_range_list, groundtruth_size= groundtruth_size,
    #     bright_n_iter= 100, bright_linear_n_iter= 100, bright_lr= None,
    #     dark_n_iter= 100, dark_linear_n_iter= 100, dark_lr= None,
    #     centre= centre, read_exsiting_csv='4_quarter_dark_DPC_NA_range[0, 1.3]_167x167_csv_file.csv',NA_range= [1.3, 1.5]
    # )

    # dark_multiplex_led_array_mask    = _simulation.dataset.get_dark_multiplex_led_array_mask(
    #         multi_angle_range= second_multi_angle_range_list, multi_radius_range= second_multi_radius_range_list, show_angle_map= False)
    # reconstruction_size = _simulation.dataset.get_reconstruction_size(multiplex_led_array_mask= dark_multiplex_led_array_mask)
    # np.savetxt('/home/kshen/Ptychography_Project/phase-retrieval-library/phase-retrieval-library/dataset/Simulation/4_quarter_dark_DPC_'
    #                    + 'NA_range[1.3, 1.5]_'+ str(reconstruction_size) +'x'+ str(reconstruction_size) +'_csv_file.csv',x_est.get(), delimiter=',')
    
    # # ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    
    # third_multi_angle_range_list  = [[0,90],[90,180],[180,270],[270,360],
    #                                   [0,90],[90,180],[180,270],[270,360],
    #                                   [0,90],[90,180],[180,270],[270,360]]

    # third_inner_NA = _simulation.dataset.NA * 1.5
    # third_outer_NA = _simulation.dataset.NA * 1.7
    # third_inner_dark_radius       = (third_inner_NA*_simulation.dataset.led_d_z)/np.sqrt(1-third_inner_NA**2)/_simulation.dataset.led_pitch
    # third_outer_dark_radius       = (third_outer_NA*_simulation.dataset.led_d_z)/np.sqrt(1-third_outer_NA**2)/_simulation.dataset.led_pitch
    # third_single_radius_range     = [np.sqrt(2*(third_inner_dark_radius**2)), np.sqrt(2*(third_outer_dark_radius**2))]
    # third_multi_radius_range_list = [single_radius_range,single_radius_range,single_radius_range,single_radius_range,
    #                                   second_single_radius_range,second_single_radius_range,second_single_radius_range,second_single_radius_range,
    #                                   third_single_radius_range,third_single_radius_range,third_single_radius_range,third_single_radius_range]

    # x_est = _simulation.dark_field_with_DPC(
    #     bright_field_angle_range = bright_angle_range, dark_multi_angle_range_list= third_multi_angle_range_list, dark_multi_radius_range_list= third_multi_radius_range_list, groundtruth_size= groundtruth_size,
    #     bright_n_iter= 100, bright_linear_n_iter= 100, bright_lr= None,
    #     dark_n_iter= 100, dark_linear_n_iter= 100, dark_lr= None,
    #     centre= centre, read_exsiting_csv='4_quarter_dark_DPC_NA_range[1.3, 1.5]_182x182_csv_file.csv',NA_range= [1.5, 1.7]
    # )

    # dark_multiplex_led_array_mask    = _simulation.dataset.get_dark_multiplex_led_array_mask(
    #         multi_angle_range= third_multi_angle_range_list, multi_radius_range= third_multi_radius_range_list, show_angle_map= False)
    # reconstruction_size = _simulation.dataset.get_reconstruction_size(multiplex_led_array_mask= dark_multiplex_led_array_mask)
    # np.savetxt('/home/kshen/Ptychography_Project/phase-retrieval-library/phase-retrieval-library/dataset/Simulation/4_quarter_dark_DPC_'
    #                    + 'NA_range[1.5, 1.7]_'+ str(reconstruction_size) +'x'+ str(reconstruction_size) +'_csv_file.csv',x_est.get(), delimiter=',')
    
    # # ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    
    # fourth_multi_angle_range_list  = [[0,90],[90,180],[180,270],[270,360],
    #                                   [0,90],[90,180],[180,270],[270,360],
    #                                   [0,90],[90,180],[180,270],[270,360],
    #                                   [0,90],[90,180],[180,270],[270,360]]

    # fourth_inner_NA = _simulation.dataset.NA * 1.7
    # fourth_outer_NA = _simulation.dataset.NA * 2
    # fourth_inner_dark_radius       = (fourth_inner_NA*_simulation.dataset.led_d_z)/np.sqrt(1-fourth_inner_NA**2)/_simulation.dataset.led_pitch
    # fourth_outer_dark_radius       = (fourth_outer_NA*_simulation.dataset.led_d_z)/np.sqrt(1-fourth_outer_NA**2)/_simulation.dataset.led_pitch
    # fourth_single_radius_range     = [np.sqrt(2*(fourth_inner_dark_radius**2)), np.sqrt(2*(fourth_outer_dark_radius**2))]
    # fourth_multi_radius_range_list = [single_radius_range,single_radius_range,single_radius_range,single_radius_range,
    #                                   second_single_radius_range,second_single_radius_range,second_single_radius_range,second_single_radius_range,
    #                                   third_single_radius_range,third_single_radius_range,third_single_radius_range,third_single_radius_range,
    #                                   fourth_single_radius_range,fourth_single_radius_range,fourth_single_radius_range,fourth_single_radius_range]

    # x_est = _simulation.dark_field_with_DPC(
    #     bright_field_angle_range = bright_angle_range, dark_multi_angle_range_list= fourth_multi_angle_range_list, dark_multi_radius_range_list= fourth_multi_radius_range_list, groundtruth_size= groundtruth_size,
    #     bright_n_iter= 100, bright_linear_n_iter= 100, bright_lr= None,
    #     dark_n_iter= 100, dark_linear_n_iter= 100, dark_lr= None,
    #     centre= centre, read_exsiting_csv='4_quarter_dark_DPC_NA_range[1.5, 1.7]_196x196_csv_file.csv',NA_range= [1.7, 2]
    # )

    # dark_multiplex_led_array_mask    = _simulation.dataset.get_dark_multiplex_led_array_mask(
    #         multi_angle_range= third_multi_angle_range_list, multi_radius_range= third_multi_radius_range_list, show_angle_map= False)
    # reconstruction_size = _simulation.dataset.get_reconstruction_size(multiplex_led_array_mask= dark_multiplex_led_array_mask)
    # np.savetxt('/home/kshen/Ptychography_Project/phase-retrieval-library/phase-retrieval-library/dataset/Simulation/4_quarter_dark_DPC_'
    #                    + 'NA_range[1.7, 2]_'+ str(reconstruction_size) +'x'+ str(reconstruction_size) +'_csv_file.csv',x_est.get(), delimiter=',')

    ## FPM ====================
    ## ========================
    # centre             = [50,60]
    # groundtruth_size   = 256

    # _simulation.bright_FPM(groundtruth_size= groundtruth_size, n_iter=2000, lr= 1e-2, centre= centre, amp_based_or_not= True)

    # NA_range        = [0,2]
    # desired_NA      = _simulation.dataset.NA * (NA_range[1]+0.5)
    # desired_radius  = ((desired_NA*_simulation.dataset.led_d_z)/np.sqrt(1-desired_NA**2)/_simulation.dataset.led_pitch)

    # _simulation.FPM(groundtruth_size= groundtruth_size, radius= desired_radius, n_iter=2000, lr= 1e-2, centre= centre, amp_based_or_not= True, NA_range= NA_range)

    
    # for n_iter in [500,1000,2000]:
    #     delete_file(f'_FPM/amp_based/n_iter={n_iter}')
    #     for outer_NA in [1.3,1.5,1.7,2]:
    #         NA_range        = [0,outer_NA]
    #         desired_NA      = _simulation.dataset.NA * (NA_range[1]+0.5)
    #         desired_radius  = ((desired_NA*_simulation.dataset.led_d_z)/np.sqrt(1-desired_NA**2)/_simulation.dataset.led_pitch)

    #         _simulation.FPM(groundtruth_size= groundtruth_size, radius= desired_radius, n_iter= n_iter, lr= 1e-2, centre= centre, amp_based_or_not= True, NA_range= NA_range, for_loop_or_not= True)