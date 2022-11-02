## Temporarily adding path
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
from scipy import interpolate

from pyphaseretrieve        import algos
from pyphaseretrieve        import phaseretrieval
from pyphaseretrieve.linop  import *

DAT_FILE_PATH = '/home/kshen/Ptychography_Project/Dataset/1LED/tif/' ## Local PATH


## ================================================== Dataset  ====================================================
## ================================================================================================================
class ptycho2d_Laura_dataSet(object):
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

    def experimentSetup(self):
        self.fourier_res           = self.get_fourierResolution()
        self.pupil_mask            = self.get_pupil_mask()

        self.NA_illu               = self.get_illumnationNA()
        self.reconstruct_size      = self.get_reconstruction_size()

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
class test_GD_in_ptych2d(object):
    def __init__(self) -> None:
        print('Ptycho2D test with GD Start\n======================')
        pass

    def generate_rand2d_x(self, in_dim, scale=0.5):
        x = 1 + scale*(np.random.randn(in_dim, in_dim) + 1j * np.random.randn(in_dim, in_dim))
        return x


    def model_test(self, camera_size:int, img_idx_array, n_iter:int, lr):
        print('Model test \n----------------------')
        ## 1. use experimental setup from Laura dataset
        self.ptycho_data        = ptycho2d_Laura_dataSet(camera_size)

        ## 2. ground truth x generating
        reconstruction_res      = self.ptycho_data.get_reconstruction_size()
        print(f'recontruction size: {reconstruction_res}')

        x                       = self.generate_rand2d_x(reconstruction_res)
        x_ft                    = np.fft.fft2(x, norm="ortho")
        
        ## 3. ptycho2d model create
        probe                   = self.ptycho_data.get_pupil_mask()
        _, _, shifts_pair       = self.ptycho_data.select_image(img_idx_array, load_img_or_not=False)
        self.ptycho_2d_model    = phaseretrieval.Ptychography2d(probe = probe, shifts_pair= shifts_pair, reconstruct_size= reconstruction_res)

        ## 4. base on exsiting paras, generate y
        y                       = np.abs(self.ptycho_2d_model.apply(x_ft))**2

        ## 5. GD solver
        GD_method               = algos.GradientDescent(self.ptycho_2d_model, line_search= None, acceleration=None)

        ## 6. solve the problem
        initial_est             = np.ones(shape=(reconstruction_res,reconstruction_res), dtype=np.complex128) 
        initial_est             = np.fft.fft2(initial_est, norm="ortho")

        x_est                   = GD_method.iterate(y = y, initial_est = initial_est, n_iter = n_iter, lr = lr)
        x_est                   = np.fft.ifft2(x_est, norm="ortho")

        ## 7. result
        print("Result correlation:")
        _x      = np.ravel(x)
        _x_est  = np.ravel(x_est)
        print(np.abs( (_x_est.T.conj() @ _x) /  (np.linalg.norm(_x_est)*np.linalg.norm(_x)) ))

        print("Result correlation without optimize:")
        _x_init  = np.ravel(np.fft.ifft2(initial_est, norm="ortho"))
        print(np.abs( (_x_init.T.conj() @ _x) /  (np.linalg.norm(_x_init)*np.linalg.norm(_x)) ))
    

    def real_data_test(self, camera_size:int, img_idx_array, n_iter:int, lr, centre:list= [0,0]):
        print('REAL dataset test \n----------------------')
        ## 1. use experimental setup from Laura dataset
        self.ptycho_data            = ptycho2d_Laura_dataSet(camera_size)

        ## 2. ground truth x generating
        reconstruction_res          = self.ptycho_data.reconstruct_size
        # reconstruction_res   = camera_size
        print(f'recontruction size: {reconstruction_res}')

        ## 4. ptycho2d model create
        probe                       = self.ptycho_data.get_pupil_mask()
        y, img_list, shifts_pair    = self.ptycho_data.select_image(img_idx_array, centre= centre)
        self.ptycho_2d_model        = phaseretrieval.Ptychography2d(probe= probe, shifts_pair= shifts_pair, reconstruct_size= reconstruction_res)

        ## 5. GD solver
        GD_method                   = algos.GradientDescent(self.ptycho_2d_model, line_search= None, acceleration=None)

        ## 6. solve the problem
        initial_est                 = np.ones(shape= (reconstruction_res,reconstruction_res), dtype= np.complex128)
        initial_est                 = np.fft.fft2(initial_est, norm="ortho")
        
        x_est                       = GD_method.iterate(y=y, initial_est=initial_est, n_iter = n_iter, lr = lr) 
        x_est                       = np.fft.ifft2(x_est, norm="ortho")

        ## 7. result
        # ptycho_data.crop_rendering()
        plt.figure()
        plt.imshow(img_list[int(len(img_list)/2)], cmap=cm.Greys_r)
        plt.colorbar()
        plt.title('Center Bright Field Image')

        plt.figure()
        plt.imshow(np.abs(x_est)**2, cmap=cm.Greys_r)
        plt.colorbar()
        plt.title('Intensity: Reconstructed image')

        plt.figure()
        plt.imshow(np.angle(x_est), cmap=cm.Greys_r)
        plt.colorbar()
        plt.title('Phase: Reconstruction image')

    
    def model_test_withoushifts(self, camera_size:int, n_img:int, n_iter:int, lr) -> None:
        print('Model test without shifts assignment \n----------------------')
        ## 1. use experimental setup from Laura dataset
        self.ptycho_data        = ptycho2d_Laura_dataSet(camera_size)

        ## 2. ground truth x generating
        reconstruction_res      = self.ptycho_data.get_reconstruction_size()
        print(f'recontruction size: {reconstruction_res}')

        x                       = self.generate_rand2d_x(reconstruction_res)
        x_ft                    = np.fft.fft2(x, norm="ortho")

        ## 3. ptycho2d model create
        probe                   = self.ptycho_data.pupil_mask
        self.ptycho_2d_model    = phaseretrieval.Ptychography2d(probe= probe, reconstruct_size= reconstruction_res, n_img= n_img)

        ## 4. base on exsiting paras, generate y
        y                       = np.abs(self.ptycho_2d_model.apply(x_ft))**2

        ## 5. GD solver
        GD_method               = algos.GradientDescent(self.ptycho_2d_model, line_search= None, acceleration=None)

        ## 6. solve the problem
        initial_est             = np.ones(shape=(reconstruction_res,reconstruction_res), dtype=np.complex128) 
        initial_est             = np.fft.fft2(initial_est, norm="ortho")

        x_est                   = GD_method.iterate(y= y, initial_est= initial_est, n_iter= n_iter, lr= lr)
        x_est                   = np.fft.ifft2(x_est, norm="ortho")

        ## 7. result
        print("Result correlation:")
        _x      = np.ravel(x)
        _x_est  = np.ravel(x_est)
        print(np.abs( (_x_est.T.conj() @ _x) /  (np.linalg.norm(_x_est)*np.linalg.norm(_x)) ))

        print("Result without optimize correlation:")
        _x_init  = np.ravel(np.fft.ifft2(initial_est, norm="ortho"))
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
    ## Test Start
    ptycho2d_test = test_GD_in_ptych2d()

    # Test 1: model test
    img_idx_array = np.linspace(1,293,293).astype(int)
    ptycho2d_test.model_test(camera_size= 50, img_idx_array= img_idx_array, n_iter= 500, lr= 0.1)
    
    # Test 2: real data
    # centre = [0,0]
    # img_idx_array = np.linspace(1,293,293).astype(int)
    # ptycho2d_test.real_data_test(camera_size= 256, img_idx_array= img_idx_array, centre= centre, n_iter= 50,lr= 3e-6)
    # plt.show()

    # Test 3: auto shift
    # ptycho2d_test.model_test_withoushifts(camera_size= 64, n_img= 17**2, n_iter= 1, lr= 0.045)  # 64, 17**2, 200, 0.0457
    # print(f'overlap rate: {ptycho2d_test.ptycho_2d_model.overlap_rate()}')
    
    # plt.figure()
    # plt.imshow(ptycho2d_test.ptycho_2d_model.get_probe_map(), cmap= cm.Greys_r)
    # plt.colorbar()
    # plt.show()