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
import matplotlib.animation as animation
import scipy.io

from pyphaseretrieve.linop  import *
from pyphaseretrieve        import algos
from pyphaseretrieve        import phaseretrieval
from pyphaseretrieve        import loss

DAT_FILE_PATH = '/home/kshen/Ptychography_Project/phase-retrieval-library/phase-retrieval-library/dataset/X_ray_FCC_particle/_mat/FCC_particle_FZP_11_dataset_id1.mat' ## Local PATH
UM2NM         = 1e3

## ================================================== Dataset  ====================================================
## ================================================================================================================
class FCC_dataset(object):
    def __init__(self, camera_size:int) -> None:

        self.mat_data = scipy.io.loadmat(DAT_FILE_PATH)

        self.camera_size                = camera_size
        self.original_camera_size       = 512
        self.total_probe_n              = 2347

        """length unit: [nm]"""
        self.pixel_pitch          = 75000
        self.x_FOV                = 50000
        self.y_FOV                = 30000
        self.photon_eV            = 6200
        self.distance_to_det      = 5.268e9

        self.probe                = np.array(self.mat_data['initial_probe'])
        self.probe_position       = np.array(self.mat_data['probe_positions'])  # with unit 27.43nm
        self.measured_intensities = np.array(self.mat_data['measured_intensities'])
        self.detector_mask        = np.array(self.mat_data['detector_mask'])

        self.experimentSetup()
        self.crop_measured_intensities()
        self.crop_detector_mask()
        self.crop_probe()

    def experimentSetup(self):
        self.wave_lambda                           = self.get_photon_lambda()
        self.original_pixel_resolution             = self.get_original_pixel_resolution()
        self.original_reconstruction_shape         = self.get_original_reconstruction_shape()
        self.centre_pos_nm                         = self.get_centre_position_nm()

        self.pixel_resolution                      = self.get_pixel_resolution()
        self.centre_pos_pixel                      = self.get_centre_position()

        self.total_shifts_v ,self.total_shifts_h , self.total_shifts_pair = self.get_shifts_pairs()

    def get_photon_lambda(self):
        e = 1.602176634e-19
        h = 6.62607015e-34
        c = 2.99792458e8
        wave_lambda = (h*c)/(self.photon_eV*e) # in [m]
        wave_lambda *= 1e+9
        return wave_lambda

    def get_original_pixel_resolution(self):
        f_spatial_max = (self.original_camera_size * self.pixel_pitch)/(self.wave_lambda*self.distance_to_det)
        pixel_resolution = 1/f_spatial_max
        return pixel_resolution

    def get_original_reconstruction_shape(self):
        recon_x = np.floor(np.max(self.probe_position[:,1]))
        recon_y = np.floor(np.max(self.probe_position[:,0]))
        reconstruction_shape = (int(recon_y), int(recon_x))
        return reconstruction_shape
    
    def get_centre_position_nm(self):
        x_center = self.x_FOV//2
        y_center = self.y_FOV//2
        centre_pos_nm = [x_center, y_center]
        return centre_pos_nm
    
    def get_pixel_resolution(self):
        f_spatial_max = (self.camera_size * self.pixel_pitch)/(self.wave_lambda*self.distance_to_det)
        pixel_resolution = 1/f_spatial_max
        return pixel_resolution

    def get_centre_position(self):
        x_center = self.x_FOV//2
        y_center = self.y_FOV//2
        x_center_pixel = x_center/self.pixel_resolution
        y_center_pixel = y_center/self.pixel_resolution
        centre_pos_pixel = [x_center_pixel,y_center_pixel]
        return centre_pos_pixel
    
    def get_shifts_pairs(self):
        shifts_v_nm = self.probe_position[:,0]*self.original_pixel_resolution
        shifts_h_nm = self.probe_position[:,1]*self.original_pixel_resolution
        shifts_v_idx = shifts_v_nm/self.pixel_resolution
        shifts_h_idx = shifts_h_nm/self.pixel_resolution
        shifts_v =   np.floor(shifts_v_idx - self.centre_pos_pixel[1])
        shifts_h =   -np.floor(shifts_h_idx - self.centre_pos_pixel[0])
        shifts_pairs = np.concatenate([shifts_v.reshape(self.total_probe_n,1), shifts_h.reshape(self.total_probe_n,1)],axis=1)
        return shifts_v, shifts_h, shifts_pairs

    def select_images(self, x_upper_um=25 ,x_lower_um= -25, y_upper_um= 15, y_lower_um= -15, scatter_plot:bool= False, remove_background:bool= False, concatenating_image:bool= True):
        img_idx_array = np.linspace(0, self.total_probe_n-1, self.total_probe_n)
        cat_imgIDX_shiftsPair = np.concatenate((img_idx_array.reshape(self.total_probe_n,1),self.total_shifts_pair), axis=1)
        
        x_centre = (x_upper_um + x_lower_um)/2*UM2NM //self.pixel_resolution
        y_centre = (y_upper_um + y_lower_um)/2*UM2NM //self.pixel_resolution

        crop_x_upper_boundry = x_upper_um*UM2NM
        crop_x_lower_boundry = x_lower_um*UM2NM
        crop_x_upper_mask    = (self.total_shifts_pair[:,1]*self.pixel_resolution) < crop_x_upper_boundry
        crop_x_lower_mask    = crop_x_lower_boundry < (self.total_shifts_pair[:,1]*self.pixel_resolution)
        crop_x_mask          = np.logical_and(crop_x_upper_mask, crop_x_lower_mask)

        crop_y_upper_boundry = y_upper_um*UM2NM
        crop_y_lower_boundry = y_lower_um*UM2NM
        crop_y_upper_mask    = (self.total_shifts_pair[:,0]*self.pixel_resolution) < crop_y_upper_boundry
        crop_y_lower_mask    = crop_y_lower_boundry < (self.total_shifts_pair[:,0]*self.pixel_resolution)
        crop_y_mask          = np.logical_and(crop_y_upper_mask, crop_y_lower_mask)

        crop_mask            = np.logical_and(crop_x_mask, crop_y_mask)
        
        cropped_imgIDX_shiftsPair = cat_imgIDX_shiftsPair[crop_mask,:]
        img_idx                   = cropped_imgIDX_shiftsPair[:,0]
        cropped_imgIDX_shiftsPair[:,1] = cropped_imgIDX_shiftsPair[:,1] - y_centre
        cropped_imgIDX_shiftsPair[:,2] = cropped_imgIDX_shiftsPair[:,2] - x_centre
        shifts_pairs              = cropped_imgIDX_shiftsPair[:,1:3]

        if scatter_plot:
            self.scatter_plot_rendering(x= cropped_imgIDX_shiftsPair[:,2], y= cropped_imgIDX_shiftsPair[:,1],title= 'Shifted Probe Position', x_label= 'X shifts [9.9 nm]', y_label= 'Y Shifts [14.16 nm]', file_name= 'shifted_probe_position')       
            # self.scatter_plot_animation(shifts_pairs)

        reconstruction_x_size     = int((np.max(shifts_pairs[:,1]) - np.min(shifts_pairs[:,1])))
        reconstruction_y_size     = int((np.max(shifts_pairs[:,0]) - np.min(shifts_pairs[:,0])))

        reconstruction_shape      = (reconstruction_y_size,reconstruction_x_size)

        img_list = []
        for _, keys in enumerate(cropped_imgIDX_shiftsPair):
            img = cp.array(self.measured_intensities[:,:,int(keys[0])]*self.detector_mask)
            img_list.append(img)  

        if remove_background:
            img_background = img_list[0]
            for _, img in enumerate(img_list):
                img_background = np.minimum(img_background, img)
        else:
                img_background = 0

        if concatenating_image:
            print('image concatenating')
            y = None
            for idx, _img in enumerate(img_list):
                _crop_img_demean = _img - img_background 
                if y is None:
                    y = _crop_img_demean
                else:
                    y = np.concatenate([y,_crop_img_demean], axis=0)
            print('finish concatenation')

            return y, img_list, shifts_pairs, img_idx, reconstruction_shape
        else:
            return img_list, shifts_pairs, img_idx, reconstruction_shape
    
    def crop_measured_intensities(self):
        v_size, h_size = self.measured_intensities[:,:,0].shape
        v_start = int(v_size//2 - (self.camera_size//2))
        h_start = int(h_size//2 - (self.camera_size//2))
        self.measured_intensities =  self.measured_intensities[v_start:v_start+self.camera_size, h_start:h_start+self.camera_size,:]
    
    def crop_detector_mask(self):
        v_size, h_size = self.detector_mask.shape
        v_start = int(v_size//2 - (self.camera_size//2))
        h_start = int(h_size//2 - (self.camera_size//2))
        self.detector_mask =  self.detector_mask[v_start:v_start+self.camera_size, h_start:h_start+self.camera_size]

    def crop_probe(self):
        v_size, h_size = self.probe.shape
        v_start = int(v_size//2 - (self.camera_size//2))
        h_start = int(h_size//2 - (self.camera_size//2))
        self.probe =  self.probe[v_start:v_start+self.camera_size, h_start:h_start+self.camera_size]

    # ----------------------- Render f ----------------------
    # -------------------------------------------------------
    def mat2img(self):
        for idx in range(self.measured_intensities.shape[2]):
            img = self.measured_intensities[:,:,idx]
            print(f'image {idx} loading')
            plt.imshow(img, cmap=cm.Greys_r)
            plt.colorbar()
            plt.title(f'FCC image {idx+1}')
            plt.savefig('/home/kshen/Ptychography_Project/phase-retrieval-library/phase-retrieval-library/dataset/X_ray_FCC_particle/pic/fcc_id1_{0:04}.png'.format(idx+1))
            plt.close()
    
    def probe_rendering(self) -> None:
        plt.figure()
        plt.imshow(np.abs(self.probe), cmap=cm.Greys_r)
        plt.colorbar()
        plt.title('Abs FCC probe mask')
        plt.savefig('fcc_probe_mask.png')
        plt.close()

    def image_rendering(self,img) -> None:
        plt.figure()
        plt.imshow(np.abs(img), cmap=cm.Greys_r)
        plt.colorbar()
        plt.title('Rendering image')
        plt.savefig('render_img.png')
        plt.close()

    def probe_position_rendaering(self):
        x = self.probe_position[:,1]
        y = self.probe_position[:,0]

        plt.figure()
        fig, ax = plt.subplots()
        ax.scatter(x, y, s=2)
        plt.title('Position of Probe')
        plt.xlabel('X Position [Unit Pixel]')
        plt.ylabel('Y Position [Unit Pixel]')
        plt.savefig('probe_position.png')
        plt.close()

    def scatter_plot_rendering(self, x, y, title:str= 'Test',x_label:str= 'X label', y_label:str= 'Y label', file_name:str = 'Test_img'):
        plt.figure()
        fig, ax = plt.subplots()
        ax.scatter(x, y, s=2)
        plt.title(title)
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.savefig(file_name+'.png')
        plt.close()
    
    def scatter_plot_animation(self, scatter_position, title:str= 'Test',x_label:str= 'X label', y_label:str= 'Y label', file_name:str = 'Test_ani'):        
        x = []
        y = []
        frames = []
        fig, ax = plt.subplots()
        for idx, position in enumerate(scatter_position):
            x.append(position[1])
            y.append(position[0])
            frames.append([ax.scatter(x, y, s=4, marker="o", edgecolor='black')])
        ax.set_title(title)
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        print('rendering...')
        ani = animation.ArtistAnimation(fig, frames, interval=10, blit=True,repeat_delay=1000)
        ani.save(file_name+'.gif', writer='pillow')

    def test_img(self):
        img = np.roll(self.probe,shift=(-200,200),axis=(0,1))
        plt.figure()
        plt.imshow(np.abs(img), cmap=cm.Greys_r)
        plt.colorbar()
        plt.title('Test image')
        plt.savefig('test.png')

    # ----------------------- Render f ----------------------
    # -------------------------------------------------------
## ================================================== END Dataset =================================================
## ================================================================================================================

class test_Xray_ptycho(object):
    def __init__(self) -> None:
        print('Ptycho2D test with GD Start\n======================')
        pass    

    def real_data_test(self, camera_size:int, n_iter:int, lr):
        print('REAL dataset test \n----------------------')
        ## 1. use experimental setup from Laura dataset
        self.ptycho_data            = FCC_dataset(camera_size)

        ## 3. ptycho2d model create
        probe                                                         = cp.array(self.ptycho_data.probe*self.ptycho_data.detector_mask)
        y, img_list, shifts_pairs, img_idx, reconstruct_shape         = self.ptycho_data.select_images(x_upper_um= 25, x_lower_um= 0, y_upper_um= 7.5, y_lower_um= -7.5, scatter_plot= True)
        print(f'recontruction shape: {reconstruct_shape}')

        self.ptycho_2d_model        = phaseretrieval.XRay_Ptychography2d(probe= probe, shifts_pair= shifts_pairs, reconstruct_shape= reconstruct_shape)

        ## 3. solver
        # loss_function               = loss.loss_amplitude_based(epsilon=0)
        GD_method                   = algos.GradientDescent(self.ptycho_2d_model, loss_func= None, line_search= False, acceleration= None)

        ## 6. solve the problem
        initial_est                 = cp.ones(shape= reconstruct_shape, dtype= np.complex128)
        x_est                       = GD_method.iterate(y=y, initial_est=initial_est, n_iter = n_iter, lr = lr)

        ## 7. result
        plt.figure()
        plt.imshow(np.abs(x_est.get())**2, cmap=cm.Greys_r)
        plt.colorbar()
        plt.title('Intensity: X-ray image')
        plt.savefig(f'_xray_FCC11_img_nointro/n_iter={n_iter}/Intensity_x=(0-25),y=(-7.5-7.5)_niter={n_iter},lr={lr}.png')

        plt.figure()
        plt.imshow(np.angle(x_est.get()), cmap=cm.Greys_r)
        plt.colorbar()
        plt.title('Phase: Reconstruction image')
        plt.savefig(f'_xray_FCC11_img_nointro/n_iter={n_iter}/Phase_x=(0-25),y=(-7.5-7.5)_niter={n_iter},lr={lr}.png')

        return x_est

if __name__ == '__main__':
    # dataset = FCC_dataset(512)
    # print(dataset.centre_pos_nm)
    # print(dataset.total_shifts_v)
    # print(dataset.total_shifts_h)
    # dataset.select_images()

    x_ray_test = test_Xray_ptycho()
    for _, n_iter in enumerate([100,150]):
        for lr in np.geomspace(1e-1, 1e-4, num=4):
            x_ray_test.real_data_test(camera_size= 512, n_iter= n_iter, lr= lr)
    # x_ray_test.real_data_test(camera_size= 512, n_iter= 50, lr= 1e-01)

    # plt.figure()
    # plt.imshow(np.abs(x_ray_test.ptycho_2d_model.get_probe_overlap_map().get()), cmap= cm.Greys_r)
    # plt.colorbar()
    # plt.savefig(f'overlap_img.png')