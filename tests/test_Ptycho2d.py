## Temporarily adding path
from array import array
import sys
from pathlib import Path
sys.path.append(str(Path().absolute()))

## === Test Start ===
import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.animation as animation
import matplotlib.patches as patches

from pyphaseretrieve        import algos
from pyphaseretrieve        import phaseretrieval
from pyphaseretrieve.linop  import *

DAT_FILE_PATH = '/Users/chenguanchen/Desktop/Master Thesis/Ptychography Dataset/data/1LED/tif/' ## Local PATH


## ================================================== Dataset  ====================================================
## ================================================================================================================
class ptycho2d_Laura_dataSet(object):
    def __init__(self, camera_size:int) -> None:
        ## Experiment Setup Parameters Setting (distance and length unit: um)
        # Optical system
        self.wave_lambda     = 0.6292
        self.NA              = 0.1
        # Camera system
        self.camera_size        = camera_size
        self.mag                = 8.1485
        self.camera_pixel_size  = 6.5
        self.CAMERA_H_RES       = 2560
        self.CAMERA_V_RES       = 2160
        # LED system
        self.led_pitch       = 4000
        self.led_d_z         = 67500
        self.led_dia_number  = 19

        ## generate experimental setup
        self.experimentSetup()

    def experimentSetup(self):
        self.fourier_res           = self.compute_fourierResolution()
        self.pupil_mask            = self.compute_pupil_mask()

        self.NA_illu               = self.compute_illumnationNA()
        self.reconstruct_size      = self.compute_reconstruction_size()

        self.total_shifts_h_map ,self.total_shifts_v_map , self.total_shifts_pair = self.compute_shifts()

    def compute_fourierResolution(self) -> tuple([float, float]):
        img_pixel_size         = self.camera_pixel_size/self.mag
        FoV                    = img_pixel_size*self.camera_size
        fourier_resolution     = 1/FoV

        return fourier_resolution

    def compute_pupil_mask(self):
        mid_pixel_idx         = math.floor(self.camera_size/2)
        pupil_radius          = self.NA/self.wave_lambda/self.fourier_res

        if (self.camera_size%2) == 0:
            camera_size_idx_upper = np.linspace(0, mid_pixel_idx, int(self.camera_size/2)+1)
            camera_size_idx_lower = np.linspace(mid_pixel_idx-1,1, int(self.camera_size/2)-1)
        else:
            camera_size_idx_upper = np.linspace(0, mid_pixel_idx, math.floor(self.camera_size/2)+1)
            camera_size_idx_lower = np.linspace(mid_pixel_idx-1,0, math.floor(self.camera_size/2))

        camera_size_idx = np.concatenate([camera_size_idx_upper,camera_size_idx_lower])

        temp_mask_h, temp_mask_v = np.meshgrid(camera_size_idx,camera_size_idx)

        pupil_mask, _ = cart2pol(temp_mask_h,temp_mask_v)
        pupil_mask[pupil_mask <= pupil_radius] = 1
        pupil_mask[pupil_mask >= pupil_radius] = 0

        return pupil_mask

    def compute_illumnationNA(self) -> float:
        led_r_number    = math.floor(self.led_dia_number/2)
        led_r           = led_r_number * self.led_pitch
        illuminationNA  = led_r/math.sqrt(self.led_d_z**2+led_r**2)

        return illuminationNA
        
    def compute_reconstruction_size(self) -> int:
        synthetic_NA            = self.NA_illu + self.NA
        reconstruct_dia_number  = 2*synthetic_NA/self.wave_lambda/self.fourier_res

        return math.ceil(reconstruct_dia_number)

    def compute_shifts(self, return_inner_paras:bool = False):
        led_r_number = (self.led_dia_number-1) / 2
        led_h = np.linspace(-led_r_number,led_r_number,self.led_dia_number)
        led_v = np.linspace(-led_r_number,led_r_number,self.led_dia_number)
        
        led_h_idx, led_v_idx = np.meshgrid(led_h,led_v)
        led_r_mask = np.sqrt(led_h_idx**2 + led_v_idx**2)
        led_r_mask[led_r_mask < self.led_dia_number/2] = 1
        led_r_mask[led_r_mask > self.led_dia_number/2] = 0

        led_h_d_map         = led_h_idx * self.led_pitch
        led_v_d_map         = led_v_idx * self.led_pitch
        led_d_to_sample_map = np.sqrt( (led_h_d_map)**2 + (led_v_d_map)**2 + self.led_d_z**2)

        sinTheta_h_map      = led_h_d_map/led_d_to_sample_map
        sinTheta_v_map      = led_v_d_map/led_d_to_sample_map

        shifts_h_map = np.round(sinTheta_h_map * led_r_mask / self.wave_lambda / self.fourier_res)
        shifts_v_map = np.round(sinTheta_v_map * led_r_mask / self.wave_lambda / self.fourier_res)

        n_used_led = int(np.sum(led_r_mask))
        shifts_h = shifts_h_map[led_r_mask == 1]
        shifts_v = shifts_v_map[led_r_mask == 1]
        shifts_pair = np.concatenate([shifts_v.reshape(n_used_led,1),shifts_h.reshape(n_used_led,1)],axis=1)

        if return_inner_paras:
            return sinTheta_h_map, sinTheta_v_map, led_r_mask
        else:
            return shifts_h_map, shifts_v_map, shifts_pair

    def compute_bright_field_LED(self):
        sinTheta_h_map, sinTheta_v_map, led_r_mask = self.compute_shifts(return_inner_paras = True)
        
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

        bright_field_led_map = ((led_NA_map <= 0.99 * self.NA) * led_index_map).astype(np.int)

        return bright_field_led_map

    def load_image(self, img_index_array, load_img_or_not = True):
        v_center = self.CAMERA_V_RES/2
        h_center = self.CAMERA_H_RES/2
        crop_half_size = int(self.camera_size/2)

        file_path = DAT_FILE_PATH

        shifts_pair = self.total_shifts_pair[img_index_array,0:2]

        if load_img_or_not:
            print('image loading...')
            img_list = []
            y        = None
            for i in img_index_array:
                file_name = str('ILED_{0:04}.tif'.format(i))
                img       = plt.imread(file_path + file_name)

                crop_img = img[int(v_center-crop_half_size):int(v_center+crop_half_size),int(h_center-crop_half_size):int(h_center+crop_half_size)]
                img_list.append(crop_img)

                if y is None:
                    y = crop_img
                else:
                    y = np.concatenate([y,crop_img], axis=0)
                # print(str('{:.2f}%').format((i-first_img_n)/(end_img_n-first_img_n)*100))
            print('finish loading...')

            return y, img_list, shifts_pair
        else:
            return None, None, shifts_pair
        
        
    def crop_rendering(self) -> None:
        v_center        = self.CAMERA_V_RES/2
        h_center        = self.CAMERA_H_RES/2
        crop_half_size  = int(self.camera_size/2)

        file_path = DAT_FILE_PATH
        file_name = str('ILED_0147.tif')
        img       = plt.imread(file_path + file_name)

        fig, ax = plt.subplots()
        ax.imshow(img, cmap=cm.Greys_r)
        ax.set_title('Center Bright field Image')
        rect = patches.Rectangle((h_center-crop_half_size, v_center-crop_half_size), self.camera_size, self.camera_size, linewidth=2, edgecolor='r', facecolor='none')
        ax.add_patch(rect)

    def pupil_rendering(self) -> None:
        plt.figure()
        plt.imshow(self.pupil_mask, cmap=cm.Greys_r)
        plt.title('Pupil mask')
        plt.show()

    def total_shifts_rendering(self) -> None:
        plt.figure()
        plt.imshow(self.total_shifts_h_map, cmap=cm.Greys_r)
        plt.title('Total Horizental shifts map')

        plt.figure()
        plt.imshow(self.total_shifts_v_map, cmap=cm.Greys_r)
        plt.title('Total Vertical shifts map')

        plt.show()
## ================================================ END Dataset ===================================================
## ================================================================================================================



## ================================================== Test Class ==================================================
## ================================================================================================================
class test_GD_in_ptych2d(object):
    def __init__(self) -> None:
        print('Ptycho2D test with GD Start\n======================')
        pass

    def generate_rand2d_x(self, in_dim):
        x = (np.random.randn(in_dim, in_dim) + 1j * np.random.randn(in_dim, in_dim))
        return x

    def model_test(self):
        print('Model test \n----------------------')
        ## 1. use experimental setup from Laura dataset
        camera_size = 20
        ptycho_data = ptycho2d_Laura_dataSet(camera_size)

        ## 2. ground truth x generating
        reconstruction_res = ptycho_data.reconstruct_size
        x = self.generate_rand2d_x(reconstruction_res)
        print(f'recontruction size: {reconstruction_res}')

        ## 3. ptycho2d model create
        probe              = ptycho_data.pupil_mask
        _, _, shifts_pair  = ptycho_data.load_image(np.arange(1,293), load_img_or_not=False)
        ptycho_2d_model    = phaseretrieval.Ptychography2d(probe, shifts_pair= shifts_pair, reconstruct_size=reconstruction_res)

        ## 4. base on exsiting paras, generate y
        y = np.abs(ptycho_2d_model.apply(x))**2

        ## 5. GD solver
        GD_method   = algos.GradientDescent(ptycho_2d_model, line_search= None, acceleration=None)

        ## 6. solve the problem
        initial_est = np.ones(shape=(reconstruction_res,reconstruction_res), dtype=np.complex128)  # lr = 0.01 - 0.1
        x_est = GD_method.iterate(y=y, initial_est=initial_est, n_iter = 100, lr = 0.07)

        ## 7. result
        print("Result correlation:")
        _x      = np.ravel(x)
        _x_est  = np.ravel(x_est)
        print(np.abs( (_x_est.T.conj() @ _x) /  (np.linalg.norm(_x_est)*np.linalg.norm(_x)) ))
    

    def real_data_test(self):
        print('REAL dataset test \n----------------------')
        ## 1. use experimental setup from Laura dataset
        camera_size = 20
        ptycho_data = ptycho2d_Laura_dataSet(camera_size)

        ## 2. ground truth x generating
        reconstruction_res = ptycho_data.reconstruct_size
        print(f'recontruction size: {reconstruction_res}')

        ## 3. image data selection
        # full_img_array            = np.arange(1,293)
        # img_idx_array             = full_img_array

        selected_img_array        = np.arange(117,177)
        img_idx_array             = selected_img_array
        
        # bright_field_led_map      = ptycho_data.compute_bright_field_LED()
        # bright_field_led_array    = bright_field_led_map[bright_field_led_map > 0]
        # img_idx_array             = bright_field_led_array

        ## 4. ptycho2d model create
        probe                     = ptycho_data.pupil_mask
        y, img_list, shifts_pair  = ptycho_data.load_image(img_idx_array)
        ptycho_2d_model = phaseretrieval.Ptychography2d(probe, shifts_pair= shifts_pair, reconstruct_size=reconstruction_res)

        ## 5. GD solver
        GD_method = algos.GradientDescent(ptycho_2d_model, line_search= None, acceleration=None)

        ## 6. solve the problem
        initial_est = np.ones(shape=(reconstruction_res,reconstruction_res), dtype=np.complex128)

        # initial_est = img_list[146].astype(np.complex128)
        # if (reconstruction_res-camera_size)%2 == 1:
        #     initial_img = np.pad(img_list[146],(math.floor((reconstruction_res-camera_size)/2),math.floor((reconstruction_res-camera_size)/2)+1),mode='edge')
        # else:
        #     initial_img = np.pad(img_list[146],math.floor((reconstruction_res-camera_size)/2),mode='edge')
        # initial_est = initial_img

        x_est = GD_method.iterate(y=y, initial_est=initial_est, n_iter = 100, lr = 0.000011)  # lr 20->20: 0.0000006, 20->29: 0.000011

        ## 7. result
        ptycho_data.crop_rendering()
        plt.figure()
        plt.imshow(img_list[int(len(img_list)/2)], cmap=cm.Greys_r)
        plt.title('LED 147 image')
        plt.figure()
        plt.imshow(np.abs(x_est)**2, cmap=cm.Greys_r)
        plt.title('Reconstruction image')
        plt.show()

## ================================================== END Test Class ==================================================
## ====================================================================================================================


## ============ functions ============
def cart2pol(x, y):
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return(rho, phi)

def demo_dataset():
    start_img = 120
    end_img   = 170

    file_path = DAT_FILE_PATH
    print('image loading...')
    img_list = [] 
    for i in range(start_img,end_img):
        file_name = str('ILED_{0:04}.tif'.format(i))
        img = plt.imread(file_path+file_name)
        img_list.append(img)
        print(str('{:.2f}%').format((i-start_img+1)/(end_img-start_img)*100))
    animation_show(img_list, data_range=[start_img,end_img])

def animation_show(img_list:list, data_range = None):
    number_of_img = len(img_list)
    frames = []
    fig = plt.figure()
    for idx in range(number_of_img):
        frames.append([plt.imshow(img_list[idx], cmap=cm.Greys_r, animated=True)])
    print('rendering...')
    ani = animation.ArtistAnimation(fig, frames, interval=200, blit=True,repeat_delay=10000)
    if data_range:
        range_map = {'start':data_range[0],'end':data_range[1]}
        plt.title('Data image from {start} to {end}'.format_map(range_map))
    plt.show()
## ============ functions ============



if __name__ == '__main__':
    ## Dataset showing
    # demo_dataset()

    # Laura_dataset = ptycho2d_Laura_dataSet(100)
    # Laura_dataset.compute_bright_field_LED()
    # Laura_dataset.pupil_rendering()
    # Laura_dataset.total_shifts_rendering()

    ## Test Start
    ptycho2d_test = test_GD_in_ptych2d()
    # ptycho2d_test.model_test()
    ptycho2d_test.real_data_test()