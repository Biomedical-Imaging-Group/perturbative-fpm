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
                plt.savefig(f'led_map{idx}.png')
        multiplex_led_array      = np.array(multiplex_led_list)
        return multiplex_led_array

    def multiplex_image_select(self, multiplex_led_array:np.ndarray, remove_background:bool = False):       
        h_start     = int(self.CAMERA_H_RES//2  - self.camera_size//2)
        v_start     = int(self.CAMERA_V_RES//2  - self.camera_size//2)
        crop_size   = self.camera_size
        print('image loading...')
        total_img_list = []
        for i in np.linspace(1,self.total_n_img,self.total_n_img).astype(int):
            file_name = str('ILED_{0:04}.tif'.format(i))
            img       = plt.imread(DAT_FILE_PATH + file_name)

            crop_img = img[v_start:v_start+crop_size, h_start:h_start+crop_size]
            total_img_list.append(crop_img)
        print('finish loading...')  
        total_img_array = np.uint64(np.array(total_img_list,ndmin=3,))

        image_sum = np.tensordot(multiplex_led_array, total_img_array,axes=1)
        
        if remove_background:
            img_background = total_img_list[0]
            for _, img in enumerate(total_img_list):
                img_background = np.minimum(img_background, img)
        else:
            img_background = 0

        measured_n_img  = multiplex_led_array.shape[0]
        y = None
        for idx in range(measured_n_img):
            image_sum[idx,:,:] = image_sum[idx,:,:] - np.sum(multiplex_led_array[idx,:])*img_background 
            if y is None:
                y = np.uint64(image_sum[idx,:,:])
            else:
                y = np.concatenate([y,np.uint64(image_sum[idx,:,:])], axis=0)
        return y, image_sum, total_img_list
    
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

    def FPM_img(self, camera_size, n_iter= 1, GD_n_iter= 5,lr= 1):
        print('REAL dataset test \n----------------------')
        ## 1. use experimental setup from Laura dataset
        self.ptycho_data    = ptycho2d_Laura_dataSet(camera_size)
        bright_LED_map,_    = self.ptycho_data.get_bright_field_LED_map()
        bright_LED_array    = bright_LED_map[bright_LED_map>0] 

        ## 2. ground truth x generating
        reconstruction_res          = self.ptycho_data.get_reconstruction_size()
        print(f'recontruction size: {reconstruction_res}')

        ## 4. ptycho2d model create
        img_idx_array               = bright_LED_array
        probe                       = cp.array(self.ptycho_data.get_pupil_mask())
        y, img_list, shifts_pair    = self.ptycho_data.select_image(img_idx_array, remove_background= False)
        y                           = cp.array(y)
        self.ptycho_2d_model        = phaseretrieval.FourierPtychography2d(probe= probe, shifts_pair= shifts_pair, reconstruct_size= reconstruction_res)

        sum_bright_img = np.uint64(np.zeros_like(img_list[0]))
        for idx in range(len(img_list)):
            sum_bright_img += img_list[idx]

        # print(mean_bright_img)
        ## 5. PPR solver
        initial_est             = cp.ones(shape=(reconstruction_res,reconstruction_res), dtype=np.complex128)
        initial_est             = np.fft.fft2(initial_est, norm="ortho")

        ppr_method              = algos.PerturbativePhase(self.ptycho_2d_model)
        # x_est                   = ppr_method.iterate_GD(y = y, initial_est = initial_est, n_iter = n_iter, GD_n_iter= GD_n_iter, lr = lr)
        x_est                   = ppr_method.iterate_CGD(y= y, initial_est= initial_est, n_iter= n_iter, CGD_n_iter= GD_n_iter)
        x_est                   = np.fft.ifft2(x_est, norm="ortho")

        ## 7. result
        plt.figure()
        plt.imshow(sum_bright_img, cmap=cm.Greys_r)
        plt.colorbar()
        plt.title('Sum Bright Field Image')
        plt.savefig('_recon_img/DPC_Sum Bright Field Image.png')

        plt.figure()
        plt.imshow(img_list[int(len(img_list)/2)-1], cmap=cm.Greys_r)
        plt.colorbar()
        plt.title('Center Bright Field Image')
        plt.savefig('_recon_img/DPC_Center Bright Field Image.png')

        plt.figure()
        plt.imshow(np.abs(x_est.get())**2, cmap=cm.Greys_r)
        plt.colorbar()
        plt.title('Intensity: Reconstructed image')
        plt.savefig('_recon_img/DPC_Intensity: Reconstructed image.png')

        plt.figure()
        plt.imshow(np.angle(x_est.get()), cmap=cm.Greys_r)
        plt.colorbar()
        plt.title('Phase: Reconstruction image')
        plt.savefig('_recon_img/DPC_Phase: Reconstruction image.png')
        return x_est
    
    def DPC_img(self, camera_size, angle_range:np.ndarray, n_iter= 1, GD_n_iter= 5,lr= 1):
        print('REAL dataset test \n----------------------')
        ## 1. use experimental setup from Laura dataset
        camera_size = 256
        self.ptycho_data    = ptycho2d_Laura_dataSet(camera_size)

        ## 2. ground truth x generating
        reconstruction_res          = camera_size
        print(f'recontruction size: {reconstruction_res}')

        ## 3. LED pattern select
        angle_range                     = angle_range
        multiplex_led_array             = self.ptycho_data.get_bright_field_multiplex_led_array(angle_range= angle_range, show_angle_map= True)
        y, image_sum, total_img_list    = self.ptycho_data.multiplex_image_select(multiplex_led_array= multiplex_led_array)
        y                               = cp.array(y)

        ## 4. ptycho2d model create
        probe                       = cp.array(self.ptycho_data.get_pupil_mask())
        total_shifts_pair           = self.ptycho_data.total_shifts_pair

        ## 5. initial guess
        initial_est             = cp.ones(shape=(reconstruction_res,reconstruction_res), dtype=np.complex128)
        initial_est             = np.fft.fft2(initial_est, norm="ortho")

        pr_model                = phaseretrieval.MultiplexedPhaseRetrieval(probe= probe,multiplex_led_mask= multiplex_led_array, shifts_pair= total_shifts_pair, reconstruct_size= reconstruction_res)

        ppr_method              = algos.PerturbativePhase(pr_model)
        x_est                   = ppr_method.iterate_GD(y= y, initial_est= initial_est, n_iter= n_iter, GD_n_iter= GD_n_iter, lr=lr)
        x_est                   = np.fft.ifft2(x_est, norm="ortho")

        ## 7. result
        # plt.figure()
        # plt.imshow(total_img_list[int(len(total_img_list)/2)-1], cmap=cm.Greys_r)
        # plt.colorbar()
        # plt.title('Center Bright Field Image')
        # plt.savefig('_recon_img/DPC_Center Bright Field Image.png')

        # plt.figure()
        # plt.imshow(sum_bright_img, cmap=cm.Greys_r)
        # plt.colorbar()
        # plt.title('Sum All Bright Field Image')
        # plt.savefig('_recon_img/DPC: Sum All Bright Field Image.png')

        # plt.figure()
        # plt.imshow(image_sum[0,:,:], cmap=cm.Greys_r)
        # plt.colorbar()
        # plt.title('Sum Top Bright Field Image')
        # plt.savefig('_recon_img/DPC: Sum Top Bright Field Image.png')

        # plt.figure()
        # plt.imshow(image_sum[1,:,:], cmap=cm.Greys_r)
        # plt.colorbar()
        # plt.title('Sum Bottom Bright Field Image')
        # plt.savefig('_recon_img/DPC: Sum Bottom Bright Field Image.png')


        plt.figure()
        plt.imshow(np.abs(x_est.get())**2, cmap=cm.Greys_r)
        plt.colorbar()
        plt.title('Intensity: Reconstructed image')
        plt.savefig('_recon_img/DPC_Intensity: Reconstructed image.png')

        plt.figure()
        plt.imshow(np.angle(x_est.get()), cmap=cm.Greys_r)
        plt.colorbar()
        plt.title('Phase: Reconstruction image')
        plt.savefig('_recon_img/DPC_Phase: Reconstruction image.png')
        return x_est

    def single_test(self):
        ## 1. use experimental setup from Laura dataset
        camera_size = 256
        self.ptycho_data    = ptycho2d_Laura_dataSet(camera_size)

        ## 2. ground truth x generating
        reconstruction_res          = camera_size
        print(f'recontruction size: {reconstruction_res}')

        ## 3. LED pattern select
        # angle_range                     = np.array([[0,180],[180,360]])
        angle_range                     = np.array([[0,90],[90,180],[180,270],[270,360]])
        multiplex_led_array             = self.ptycho_data.get_bright_field_multiplex_led_array(angle_range= angle_range, show_angle_map= True)
        y, image_sum, total_img_list    = self.ptycho_data.multiplex_image_select(multiplex_led_array= multiplex_led_array)
        y                               = cp.array(y)

        ## 4. ptycho2d model create
        probe                       = cp.array(self.ptycho_data.get_pupil_mask())
        total_shifts_pair           = self.ptycho_data.total_shifts_pair

        ## 5. initial guess
        initial_est             = cp.ones(shape=(reconstruction_res,reconstruction_res), dtype=np.complex128)
        initial_est             = np.fft.fft2(initial_est, norm="ortho")

        pr_model                = phaseretrieval.MultiplexedPhaseRetrieval(probe= probe,multiplex_led_mask= multiplex_led_array, shifts_pair= total_shifts_pair, reconstruct_size= reconstruction_res)

        ppr_method              = algos.PerturbativePhase(pr_model)
        x_est                   = ppr_method.iterate_GD(y= y, initial_est= initial_est, n_iter= 1, GD_n_iter= 15, lr=1e-2)
        x_est                   = np.fft.ifft2(x_est, norm="ortho")

        ## 6. Forward model create
        # op_ifft2        = LinOpIFFT2() 
        # op_fftshift     = LinOpFFTSHIFT()
        # op_ifftshift    = LinOpIFFTSHIFT()
        # op_fcrop        = LinOpCrop2(reconstruction_res, probe.shape[0])
        # op_probe        = LinOpMul(probe)

        # total_linop_list = []
        # for _, i_mask in enumerate(multiplex_led_array):
        #     shifts_pair = total_shifts_pair[i_mask,:]
        #     linop_list = []
        #     for _, shifts in enumerate(shifts_pair):
        #         _linop = op_ifft2 @ op_probe @ op_ifftshift @ op_fcrop @ op_fftshift @ LinOpRoll2(shifts[0],shifts[1])
        #         linop_list.append(_linop)
        #     total_linop_list.append(linop_list)
        # total_linop_array = np.array(total_linop_list)
        # linop_list        = np.sum(total_linop_array,axis=1)

        ## 7. iteration Start
        # x_est                   = initial_est

        # y_list         = []
        # for _, i_array in enumerate(total_linop_array):
        #     single_y         = 0
        #     for i_linop in i_array:
        #         single_y         += np.abs(i_linop.apply(x_est))**2
        #     y_list.append(single_y)
        # y_array         = cp.array(y_list)
        
        # y_est     = None
        # for idx in range(y_array.shape[0]):
        #     if y_est is None:
        #         y_est = y_array[idx,:,:]  
        #     else:
        #         y_est = np.concatenate((y_est,y_array[idx,:,:]), axis=0)

        # perturbative_model_list = []
        # for _, i_array in enumerate(total_linop_array):
        #     _perturbative_model = None
        #     for i_linop in i_array:
        #         _out_field = i_linop.apply(x_est)
        #         if _perturbative_model is None:
        #             _perturbative_model = 2 * LinOpReal() @ LinOpMul(_out_field.conj()) @ i_linop
        #         else:
        #             _perturbative_model += 2 * LinOpReal() @ LinOpMul(_out_field.conj()) @ i_linop
        #     perturbative_model_list.append(_perturbative_model)
        # perturbative_model = StackLinOp(perturbative_model_list)

        # epsilon = np.zeros_like(x_est)
        # for gd_i_iter in range(50):
        #     grad = (-2 * perturbative_model.applyT(y - y_est - perturbative_model.apply(epsilon)))                   
        #     epsilon = epsilon - (1e-7)*grad
        #     print(gd_i_iter+1)

        # x_est += epsilon
        # x_est  = np.fft.ifft2(x_est, norm="ortho")
        ## 8. Result rendering
        plt.figure()
        plt.imshow(np.abs(x_est.get())**2, cmap=cm.Greys_r)
        plt.colorbar()
        plt.title('Intensity: Reconstructed image')
        plt.savefig('_recon_img/Single_test_DPC_Intensity: Reconstructed image.png')

        plt.figure()
        plt.imshow(np.angle(x_est.get()), cmap=cm.Greys_r)
        plt.colorbar()
        plt.title('Phase: Reconstruction image')
        plt.savefig('_recon_img/Single_test_DPC_Phase: Reconstruction image.png')


## ================================================== END Test Class ==================================================
## ====================================================================================================================


## ============ functions ============
def cart2pol(x, y):
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)/math.pi * 180
    return(rho, phi)
## ============ functions ============



if __name__ == '__main__':
    # dataset = ptycho2d_Laura_dataSet(256)
    # angle_range = np.array([[0,180],[180,360]])
    # multiplex_array = dataset.get_bright_field_multiplex_led_array(angle_range,show_angle_map=True)
    # print(multiplex_array.shape)

    # multiplex_array = np.array([[1,0,0,0],[1,0,0,1]])
    # y = dataset.multiplex_image_select(multiplex_led_array= multiplex_array)
    # print(y.shape)

    DPC_obj = DPC_test()
    # 1. FPM No multiplex
    # DPC_obj.FPM_img(camera_size= 256, n_iter= 1, GD_n_iter= 20,lr= None)
    # DPC_obj.FPM_img(camera_size= 256, n_iter= 1, GD_n_iter= 200,lr= 1e-3)

    ## 2. DPC multiplex
    angle_range = np.array([[0,180],[180,360]])
    # angle_range = np.array([[0,90],[90,180],[180,270],[270,360]])
    DPC_obj.DPC_img(camera_size= 256, angle_range= angle_range, n_iter= 1, GD_n_iter= 15,lr= 1e-7)

    # DPC_obj.single_test()
    