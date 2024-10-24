from pyphaseretrieve import phaseretrieval
from pyphaseretrieve import algos
from pyphaseretrieve.linop import *
import skimage.io as skio
from scipy import ndimage
import re
import csv
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import json
import math
import cupy as cp
import numpy as np
import os

X500_RESOLUTION_IMAGE_PATH = '/home/kshen/Ptychography_Project/phase-retrieval-library/phase-retrieval-library/dataset/V2_new_phase_target/0.25NA_500ms_10X/custom_500ms_centre[-290,-230]_camerasize256.tif'
X500_RESOLUTION_BACKGROUND_PATH = '/home/kshen/Ptychography_Project/phase-retrieval-library/phase-retrieval-library/dataset/V2_new_phase_target/Background_0.25NA_500ms_10X/custom_50ms_centre[-290,-230]_background_camerasize256.tif'

X50_RESOLUTION_IMAGE_PATH = '/home/kshen/Ptychography_Project/phase-retrieval-library/phase-retrieval-library/dataset/V2_new_phase_target/0.25NA_50ms_10X/custom_50ms_centre[-290,-230]_camerasize256.tif'
X50_RESOLUTION_BACKGROUND_PATH = '/home/kshen/Ptychography_Project/phase-retrieval-library/phase-retrieval-library/dataset/V2_new_phase_target/Background_0.25NA_50ms_10X/custom_50ms_centre[-290,-230]_background_camerasize256.tif'

FPM_SINGLE_BACKGROUND_IMG_FILE_PATH = '/home/kshen/Ptychography_Project/phase-retrieval-library/phase-retrieval-library/dataset/V2_new_phase_target/0.25NA_1000ms_10X/custom_FPM_centre[-290,-230]_back_ground_camerasize256.tif'

LED_POS_NA_FILE_PATH = '/home/kshen/Ptychography_Project/phase-retrieval-library/phase-retrieval-library/dataset/new_phase_target/led_array_pos_na_z65mm.json'  # Local PATH
LED_PATTERNS_FILE_PATH = '/home/kshen/Ptychography_Project/phase-retrieval-library/phase-retrieval-library/dataset/V2_new_phase_target/new_led_patterns.json'


# ================================================== Dataset  ====================================================
# ================================================================================================================
class new_custom_dataSet(object):
    def __init__(self, camera_size: int) -> None:
        # Experiment Setup Parameters Setting (distance and length unit: um)
        # Optical system
        self.wave_lambda = 0.525
        self.NA = 0.25
        # Camera system
        self.camera_size = camera_size
        self.mag = 10
        self.camera_pixel_Gsize = 6.5
        self.CAMERA_H_RES = 2160
        self.CAMERA_V_RES = 2560
        # LED system
        # self.led_pitch          = 4000
        # self.led_d_z            = 67500
        self.led_dia_number = 19
        self.total_LED_num = 245
        # generate experimental setup
        self.experimentSetup()

    # Initialize experimental parameters
    def experimentSetup(self):
        with open(LED_POS_NA_FILE_PATH, 'r') as open_file:
            self.led_pos_na_dict = json.load(open_file)
        with open(LED_PATTERNS_FILE_PATH, 'r') as open_file:
            self.led_patterns_dict = json.load(open_file)

        self.fourier_res = self.get_fourierResolution()
        self.pupil_mask = self.get_pupil_mask()
        self.led_total_na_dict = self.get_led_total_na_dict()
        self.led_na_dict = self.get_led_na_dict()
        self.total_shifts_pair = self.get_shiftsPairs()

    # ----------------------- Default setup -----------------------
    def get_fourierResolution(self) -> float:
        img_pixel_size = self.camera_pixel_Gsize/self.mag
        FoV = img_pixel_size*self.camera_size
        fourier_resolution = 1/FoV

        if 2*self.NA/self.wave_lambda > (1/img_pixel_size)/2:
            print('ALIASING HAPPEN!')
        return fourier_resolution

    def get_pupil_mask(self):
        pupil_radius = self.NA/self.wave_lambda/self.fourier_res

        camera_size_idx = np.linspace(-math.floor(self.camera_size/2),
                                      math.ceil(self.camera_size/2)-1, self.camera_size)
        temp_mask_h, temp_mask_v = np.meshgrid(
            camera_size_idx, camera_size_idx)

        pupil_mask, _ = cart2pol(temp_mask_h, temp_mask_v)
        pupil_mask[pupil_mask <= pupil_radius] = 1
        pupil_mask[pupil_mask >= pupil_radius] = 0
        return np.fft.fftshift(pupil_mask)
    # ----------------------- Read LED pos and NA -------------------------

    def get_led_total_na_dict(self) -> dict:
        total_na_dict = {}
        for idx, (key, value) in enumerate(self.led_pos_na_dict['led_position_list_na'].items()):
            total_na_dict[key] = np.sqrt(value[0]**2 + value[1]**2)
            if int(key) == self.total_LED_num:
                break
        return total_na_dict

    def get_led_na_dict(self) -> dict:
        led_na_dict = {}
        for idx, (key, value) in enumerate(self.led_pos_na_dict['led_position_list_na'].items()):
            led_na_dict[key] = value
            if int(key) == self.total_LED_num:
                break
        return led_na_dict
    # ----------------------- NA and reconstruction -----------------------

    def get_reconstruction_size(self, total_selected_led_idx_array: np.ndarray) -> int:
        if total_selected_led_idx_array is not None:
            NA_illu = self.get_illumnationNA_by_selected_index(
                total_selected_led_idx_array=total_selected_led_idx_array)
        else:
            raise NameError('Incorrect Input')
        synthetic_NA = NA_illu + self.NA
        max_reconstruction_size = math.floor(
            2*synthetic_NA/self.wave_lambda/self.fourier_res)
        reconstruction_size = np.maximum(
            self.camera_size, max_reconstruction_size)

        print('illumination NA: {:.2f}'.format(NA_illu))
        print('Total NA: {:.2f}'.format(synthetic_NA))
        return reconstruction_size

    def get_illumnationNA_by_selected_index(self, total_selected_led_idx_array: np.ndarray):
        temp_na_list = []
        for item in total_selected_led_idx_array:
            temp_na_list.append(self.led_total_na_dict[str(item)])

        illuminationNA = np.max(temp_na_list)
        return illuminationNA
    # ----------------------- Total shifts and shifts pairs -----------------------

    def get_shiftsPairs(self):
        shifts_h_list = []
        shifts_v_list = []
        for led_idx in range(self.total_LED_num):
            shifts_h_list.append(self.led_na_dict[str(
                led_idx)][0] / self.wave_lambda / self.fourier_res)
            shifts_v_list.append(self.led_na_dict[str(
                led_idx)][1] / self.wave_lambda / self.fourier_res)
            # CHANGE!
        shifts_h = np.array(shifts_h_list)
        shifts_v = np.array(shifts_v_list)
        shifts_pair = np.concatenate([shifts_v.reshape(
            self.total_LED_num, 1), shifts_h.reshape(self.total_LED_num, 1)], axis=1)
        return shifts_pair
    # ----------------------- Measurement selection methods -----------------------

    def selectImg_by_singleDAT_selectImgList(self, img_num_array, centre: list = [0, 0]) -> np.ndarray:
        print('image loading...')
        pattern = '0.25NA'

        image_size = self.camera_size
        h_start = int(image_size//2 + centre[0] - self.camera_size//2)
        v_start = int(image_size//2 - centre[1] - self.camera_size//2)
        crop_size = self.camera_size

        img_bright = skio.imread(X50_RESOLUTION_IMAGE_PATH)
        img_dark = skio.imread(X500_RESOLUTION_IMAGE_PATH)

        img_background_bright = skio.imread(X50_RESOLUTION_BACKGROUND_PATH)
        img_background_dark = skio.imread(X500_RESOLUTION_BACKGROUND_PATH)

        FPM_img_background = skio.imread(FPM_SINGLE_BACKGROUND_IMG_FILE_PATH)

        y = None
        for image_key in img_num_array:
            if image_key < 3:
                img = img_bright
                background_fac = 50/1000
                img_background = img_background_bright
            else:
                img = img_dark/10
                background_fac = (500/1000)/10
                img_background = img_background_dark

            constant_background = len(
                self.led_patterns_dict[pattern][image_key]) * np.mean(FPM_img_background) * background_fac
            print(f'Background value: {constant_background}')

            # single_img                                      = (img[image_key][v_start:v_start+crop_size, h_start:h_start+crop_size] - constant_background)
            # single_img                                      = (img[image_key][v_start:v_start+crop_size, h_start:h_start+crop_size] - img_background[image_key]) + ndimage.median_filter(img_background[image_key],20)
            # if image_key == 2:
            #     single_img                                      = (img[image_key][v_start:v_start+crop_size, h_start:h_start+crop_size] - img_background[image_key]) + ndimage.median_filter(img_background[image_key],20)
            # else:
            single_img = (img[image_key][v_start:v_start+crop_size, h_start:h_start+crop_size] -
                          img_background[image_key]) + ndimage.median_filter(img_background[image_key], 20)

            # single_img                                      = (img[image_key][v_start:v_start+crop_size, h_start:h_start+crop_size] - img_background[image_key]) + np.min(img_background[image_key])
            single_img = (single_img - constant_background)
            single_img[single_img < 0] = 0
            single_img_999_value = np.percentile(single_img, 99.9)
            single_img[single_img >
                       single_img_999_value] = single_img_999_value

            plt.figure()
            plt.imshow(single_img, cmap=cm.Greys_r)
            plt.colorbar()
            plt.title(f'measurement {image_key}')
            plt.savefig(f'measurement {image_key}.png')
            plt.close()

            plt.figure()
            plt.imshow(img_background[image_key], cmap=cm.Greys_r)
            plt.colorbar()
            plt.title(f'img_background {image_key}')
            plt.savefig(f'img_background {image_key}.png')
            plt.close()

            if y is None:
                y = np.uint64(single_img)
            else:
                y = np.concatenate([y, np.uint64(single_img)], axis=0)
        print('finish loading...')
        return y

    def selectImg_2_multiplexMask_totalSelectArray(self, img_num_array, led_map_fig_or_not=True):
        pattern = '0.25NA'

        total_selected_led_idx_list = []
        multiplex_mask_list = []
        for idx_num, img_idx in enumerate(img_num_array):
            multiplex_mask_list.append(self.single_selectImg2multiplex_array(
                selectImg_list=self.led_patterns_dict[pattern][img_idx]))
            total_selected_led_idx_list = total_selected_led_idx_list + \
                self.led_patterns_dict[pattern][img_idx]

            if led_map_fig_or_not:
                x_pos = []
                y_pos = []

                for selected_led in self.led_patterns_dict[pattern][img_idx]:
                    x_pos.append(
                        self.led_pos_na_dict['led_position_list_cartesian'][str(selected_led)][0])
                    y_pos.append(
                        self.led_pos_na_dict['led_position_list_cartesian'][str(selected_led)][1])

                plt.figure()
                plt.scatter(x=x_pos, y=y_pos)
                plt.colorbar()
                plt.title(f'LED pattern {idx_num+1}')
                plt.xlim(-31, 31)
                plt.ylim(-31, 31)
                plt.savefig(f'led_pattern/LED pattern {idx_num+1}')
                plt.close()

        multiplex_mask = np.array(multiplex_mask_list)

        return multiplex_mask, total_selected_led_idx_list

    def single_selectImg2multiplex_array(self, selectImg_list: list) -> np.ndarray:
        multiplex_array = np.zeros((self.total_LED_num,))
        for idx in range(self.total_LED_num):
            if idx in np.array(selectImg_list):
                multiplex_array[idx] = 1
        return multiplex_array

# ================================================ END Dataset ===================================================
# ================================================================================================================

# ============ functions ============
# ===================================


def cart2pol(x, y):
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)/np.pi * 180
    phi = (phi + 360) % 360
    return (rho, phi)


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
                img = np.zeros((len(img_row), len(img_row))
                               ).astype(np.complex128)
            for idx_col, img_element in enumerate(img_row):
                num_of_element = re.findall(
                    "[-]\d+[.]\d+|\d+[.]\d+", img_element)
                power_of_element = re.findall("e[+,-]\d+", img_element)

                img[idx_row, idx_col] = float(num_of_element[0])*float('1'+power_of_element[0]) + 1j*(
                    float(num_of_element[1])*float('1'+power_of_element[1]))

    return img
# ============ functions ============
# ===================================


class PPR_DPC_solver(object):
    def __init__(self, camera_size: int) -> None:
        self.camera_size = camera_size
        self.dataset = new_custom_dataSet(camera_size=camera_size)

    def Real_DPC(self, selectedImg_list: list,  centre: list = [0, 0], alpha=0, file_pattern='2 halves',
                 initial_est: np.ndarray = None, for_loop_or_not: bool = False, NA_range: list = [0, 1]):

        multiplex_led_array_mask, total_selected_led_idx_array = self.dataset.selectImg_2_multiplexMask_totalSelectArray(
            img_num_array=selectedImg_list)
        reconstruction_res = self.dataset.get_reconstruction_size(
            total_selected_led_idx_array=total_selected_led_idx_array)
        reconstruction_shape = (reconstruction_res, reconstruction_res)
        print(f'Real DPC recontruction shape: {reconstruction_shape}')

        DPC_y = np.array(self.dataset.selectImg_by_singleDAT_selectImgList(
            img_num_array=selectedImg_list, centre=centre))
        probe = np.array(self.dataset.get_pupil_mask())
        total_shifts_pair = self.dataset.total_shifts_pair

        total_image = int(DPC_y.shape[0]/self.camera_size)
        DPC_y_list = []
        for idx_img in range(total_image):
            DPC_y_list.append(
                DPC_y[idx_img*self.camera_size:(idx_img+1)*self.camera_size, :])

        croplinop = LinOpCrop2(in_shape=(reconstruction_shape), crop_shape=(
            self.camera_size, self.camera_size))
        probe = np.fft.ifftshift(croplinop.applyT(np.fft.fftshift(probe)))

        transfer_func_list = []
        for i_mask in multiplex_led_array_mask:
            transfer_func = np.zeros_like(probe)
            for idx, mask_item in enumerate(i_mask):
                if (mask_item != False) and (mask_item != 0):
                    pos_shift = LinOpRoll2(
                        total_shifts_pair[idx, 0], total_shifts_pair[idx, 1])
                    neg_shift = LinOpRoll2(
                        -total_shifts_pair[idx, 0], -total_shifts_pair[idx, 1])

                    if transfer_func is None:
                        transfer_func = np.copy(neg_shift.apply(
                            probe) - pos_shift.apply(probe))
                    else:
                        transfer_func = transfer_func + \
                            neg_shift.apply(probe) - pos_shift.apply(probe)
            transfer_func_list.append(transfer_func)

        file_path = str(f'_recon_img/')
        file_header = str('DPC_transfor')
        file_pattern = str('pattern: ' + file_pattern)
        file_end = str('.png')

        sum_transfer_DPC_phase = np.zeros_like(probe)
        sum_transfer_func_square = np.zeros_like(probe)
        for idx, _transfer in enumerate(transfer_func_list):
            FT_DPC_y = np.fft.fft2(DPC_y_list[idx], norm='ortho')
            FT_DPC_y = np.fft.ifftshift(croplinop.applyT(
                np.fft.fftshift(FT_DPC_y))) * reconstruction_res/self.camera_size

            sum_transfer_DPC_phase = sum_transfer_DPC_phase + \
                (_transfer * 1j).conj() * FT_DPC_y
            sum_transfer_func_square = sum_transfer_func_square + \
                np.abs(_transfer)**2

            plt.figure()
            plt.imshow(np.fft.fftshift(_transfer), cmap=cm.Greys_r)
            plt.colorbar()
            plt.title(file_header + f' function{idx}, ' + file_pattern)
            plt.savefig(file_path + file_header +
                        f' function {idx} ' + file_pattern + file_end)
            plt.close()

        FT_phase = sum_transfer_DPC_phase/(sum_transfer_func_square + alpha)
        x_est_phase = np.fft.ifft2(FT_phase, norm='ortho')

        file_alpha = str(f'alpha={alpha}')
        plt.figure()
        plt.imshow(np.real(x_est_phase), cmap=cm.Greys_r)
        plt.colorbar()
        plt.title(file_header + f' phase, ' + file_pattern)
        plt.savefig(file_path + file_header + ' phase, ' +
                    file_alpha + f', ' + file_pattern + file_end)
        plt.close()

        croplinop = LinOpCrop2(in_shape=(
            self.camera_size*2, self.camera_size*2), crop_shape=reconstruction_shape)
        file_iter = str(f'alpha={alpha}')
        padded_phase_FT = croplinop.applyT(np.fft.fftshift(
            np.fft.fft2(np.real(x_est_phase), norm='ortho')))
        padded_phase_img = np.real(np.fft.ifft2(np.fft.ifftshift(
            padded_phase_FT), norm='ortho') * (self.camera_size*2)/reconstruction_shape[0])
        img_size = self.camera_size*2
        line_x_pos = 83
        line_y_pos = [132, 141]
        y_width = line_y_pos[1] - line_y_pos[0]
        plot_x = int((line_x_pos/self.camera_size)*img_size)
        plot_y = [int((line_y_pos[0]/self.camera_size)*img_size),
                  int((line_y_pos[1]/self.camera_size)*img_size)]

        plt.figure()
        plt.imshow(padded_phase_img, cmap=cm.Greys_r)
        plt.plot([plot_x, plot_x], [plot_y[0], plot_y[1]], 'r-', lw=1.5)
        plt.colorbar()
        plt.title(f'Phase: ' + file_iter)
        plt.savefig(file_path + file_header +
                    'Phase: with r line: ' + file_iter + file_end)
        plt.close()

        plt.figure()
        plt.imshow(padded_phase_img[plot_y[0]:plot_y[1],
                   plot_x-y_width:plot_x+y_width], cmap=cm.Greys_r)
        plt.plot([y_width, y_width], [0, 2*y_width-1], 'r-', lw=1.5)
        plt.colorbar()
        plt.title(f'Phase: Zoom in ' + file_iter)
        plt.savefig(file_path + file_header +
                    'Phase: Zoom in with r line: ' + file_iter + file_end)
        plt.close()

        y = padded_phase_img[plot_y[0]:plot_y[1], plot_x]
        x = np.linspace(0, y.shape[0], y.shape[0])
        plt.figure()
        plt.plot(x, y, 'r-', lw=1.5)
        plt.title(f'Profile of cross-section: ' + file_iter)
        plt.savefig(file_path + file_header +
                    'Profile of cross-section: ' + file_iter + file_end)
        plt.close()

    def PPR_DPC(self, selectedImg_list: list, n_iter=1, linear_n_iter=5, lr=1, centre: list = [0, 0], initial_est: np.ndarray = None,
                for_loop_or_not: bool = False, NA_range: list = [0, 1],
                _lambda=0, alpha=0):
        print('PPR-DPC start \n----------------------')

        multiplex_led_array_mask, total_selected_led_idx_array = self.dataset.selectImg_2_multiplexMask_totalSelectArray(
            img_num_array=selectedImg_list)

        reconstruction_res = self.dataset.get_reconstruction_size(
            total_selected_led_idx_array=total_selected_led_idx_array)
        reconstruction_shape = (reconstruction_res, reconstruction_res)

        print(f'Recontruction shape: {reconstruction_shape}')

        y = cp.array(self.dataset.selectImg_by_singleDAT_selectImgList(
            img_num_array=selectedImg_list))
        probe = cp.array(self.dataset.get_pupil_mask())
        total_shifts_pair = self.dataset.total_shifts_pair
        pr_model = phaseretrieval.MultiplexedPhaseRetrieval(
            probe=probe, multiplex_led_mask=multiplex_led_array_mask, shifts_pair=total_shifts_pair, reconstruct_shape=reconstruction_shape)

        if initial_est is None:
            initial_est = cp.ones(
                shape=reconstruction_shape, dtype=np.complex128)
        else:
            initial_est = cp.array(initial_est)
        initial_est = np.fft.fft2(initial_est, norm="ortho")

        y_est = pr_model.apply_ModularSquare(initial_est)
        mean_y_est = np.mean(y_est[0:self.camera_size, :])
        mean_y = np.mean(y[0:self.camera_size, :])
        normalized_facter = np.sqrt(mean_y/mean_y_est)
        initial_est = initial_est * normalized_facter
        print(f'normalize factor: {normalized_facter}')

        crop_op = LinOpCrop2(in_shape=reconstruction_shape,
                             crop_shape=initial_est.shape)
        initial_est = np.fft.ifftshift(
            crop_op.applyT(np.fft.fftshift(initial_est)))

        ppr_method = algos.PerturbativePhase(pr_model)

        if lr is not None:
            x_est = ppr_method.iterate_GradientDescent(
                y=y, initial_est=initial_est, n_iter=n_iter, linear_n_iter=linear_n_iter, lr=lr, alpha=alpha)
        else:
            x_est = ppr_method.iterate_ConjugateGradientDescent(
                y=y, initial_est=initial_est, n_iter=n_iter, linear_n_iter=linear_n_iter, _lambda=_lambda)

        x_est = np.fft.ifft2(x_est, norm="ortho")

        if for_loop_or_not:
            if lr is None:
                file_path = str(f'_PPR_DPC/CGD/n_iter={n_iter}')
            else:
                file_path = str(f'_PPR_DPC/GD/n_iter={n_iter}')
        else:
            file_path = str(f'_recon_img')
        file_header = str('/ResolutionImage_PPR-DPC_')
        if lr is None:
            file_iter = str(f'n_iter={n_iter}, l_n_iter={linear_n_iter}, lambda = {
                            _lambda}, NA_range[{NA_range[0]}, {NA_range[1]}]')
        else:
            file_iter = str(f'n_iter={n_iter}, l_n_iter={linear_n_iter}, lr={
                            lr}, NA_range[{NA_range[0]}, {NA_range[1]}]')
        file_end = str('.png')

        plt.figure()
        plt.imshow(np.abs(x_est.get())**2, cmap=cm.Greys_r)
        plt.colorbar()
        plt.title(f'Intensity: ' + file_iter)
        plt.savefig(file_path + file_header +
                    'Intensity: ' + file_iter + file_end)
        plt.close()

        plt.figure()
        plt.imshow(np.angle(x_est.get()), cmap=cm.Greys_r)
        plt.colorbar()
        plt.title(f'Phase: ' + file_iter)
        plt.savefig(file_path + file_header + 'Phase: ' + file_iter + file_end)
        plt.close()

        croplinop = LinOpCrop2(in_shape=(
            self.camera_size*2, self.camera_size*2), crop_shape=reconstruction_shape)
        img = np.abs(
            (np.fft.fftshift(np.fft.fft2(np.angle(x_est.get()), norm='ortho'))))
        img = np.log(img)
        img = croplinop.applyT(img)

        NA_radius = self.dataset.NA/self.dataset.wave_lambda/self.dataset.fourier_res

        _, axes = plt.subplots()
        plt_circle = plt.Circle(
            (self.camera_size, self.camera_size), NA_radius, fill=False, color='r')
        axes.set_aspect(1)
        axes.add_artist(plt_circle)
        plt.imshow(img, cmap=cm.Greys_r)
        plt.colorbar()
        axes.set_title('Log Abs of FT Phase image ' + file_iter)
        plt.savefig(file_path + file_header +
                    'Phase: Log FT image ' + file_iter + file_end)
        plt.close()

        phase_img = np.angle(x_est)
        phase_img_99 = np.percentile(np.abs(phase_img), 99)
        phase_img[phase_img > phase_img_99] = 0
        phase_img[phase_img < -phase_img_99] = 0
        plt.figure()
        plt.imshow(phase_img.get(), cmap=cm.Greys_r)
        plt.colorbar()
        plt.title(f'Phase Image, filter 1%: ' + file_iter)
        plt.savefig(file_path + file_header +
                    'Phase99: ' + file_iter + file_end)
        plt.close()

        padded_phase_FT = croplinop.applyT(
            np.fft.fftshift(np.fft.fft2(x_est, norm='ortho')))
        padded_phase_img = np.fft.ifft2(np.fft.ifftshift(
            padded_phase_FT), norm='ortho') * (self.camera_size*2)/reconstruction_shape[0]
        img_size = self.camera_size*2
        line_x_pos = 83
        line_y_pos = [132, 141]
        y_width = line_y_pos[1] - line_y_pos[0]
        plot_x = int((line_x_pos/self.camera_size)*img_size)
        plot_y = [int((line_y_pos[0]/self.camera_size)*img_size),
                  int((line_y_pos[1]/self.camera_size)*img_size)]

        plt.figure()
        plt.imshow(np.angle(padded_phase_img.get()), cmap=cm.Greys_r)
        plt.plot([plot_x, plot_x], [plot_y[0], plot_y[1]], 'r-', lw=1.5)
        plt.colorbar()
        plt.title(f'Phase: ' + file_iter)
        plt.savefig(file_path + file_header +
                    'Phase: with r line: ' + file_iter + file_end)
        plt.close()

        plt.figure()
        plt.imshow(np.angle(padded_phase_img.get()[
                   plot_y[0]:plot_y[1], plot_x-y_width:plot_x+y_width]), cmap=cm.Greys_r)
        plt.plot([y_width, y_width], [0, 2*y_width-1], 'r-', lw=1.5)
        plt.colorbar()
        plt.title(f'Phase: Zoom in ' + file_iter)
        plt.savefig(file_path + file_header +
                    'Phase: Zoom in with r line: ' + file_iter + file_end)
        plt.close()

        y = np.angle(padded_phase_img.get()[plot_y[0]:plot_y[1], plot_x])
        x = np.linspace(0, y.shape[0], y.shape[0])
        plt.figure()
        plt.plot(x, y, 'r-', lw=1.5)
        plt.title(f'Profile of cross-section: ' + file_iter)
        plt.savefig(file_path + file_header +
                    'Profile of cross-section: ' + file_iter + file_end)
        plt.close()

        return x_est

    def auto_CGD_for_loop(self, selectedImg_list, NA_range, centre=[0, 0], initial_est=None, _lambda=0, alpha=0):
        # for n_iter in [1,2,4,6,8,10,15]:
        for n_iter in [1, 2, 3, 4, 5, 6, 7, 8]:
            # delete_file(f'_PPR_DPC/CGD/n_iter={n_iter}')
            for linear_n_iter in [1, 2, 3, 4, 5, 7, 8, 9, 10, 20, 30, 50, 100]:
                # for linear_n_iter in [10,50,100]:
                self.PPR_DPC(selectedImg_list=selectedImg_list,
                             n_iter=n_iter, linear_n_iter=linear_n_iter, lr=None,
                             initial_est=initial_est,
                             centre=centre, for_loop_or_not=True, NA_range=NA_range,
                             _lambda=_lambda, alpha=alpha)


if __name__ == '__main__':
    # clean folder
    delete_file('led_pattern')
    delete_file('_recon_img')
    # delete_file('_recon_img_temp')

    # np.savetxt('/home/kshen/Ptychography_Project/phase-retrieval-library/phase-retrieval-library/dataset/V2_new_phase_target/3_patterns_dark_[1,1.6]_PPR_DPC_niter=4_lniter=3_lambda=2400_csv_file.csv',initial_est, delimiter=',')

    # centre          = [-290,-230]
    _PPR_DPC_solver = PPR_DPC_solver(camera_size=256)

    # Real DPC ==================================================
    # img_list    = [0,1]
    # _PPR_DPC_solver.Real_DPC(selectedImg_list= img_list, alpha= 1)

    # Bright PPR ==================================================
    # img_list    = [0,1,2]
    # for _lambda in [1,10,100,300,500,800]:
    #     _PPR_DPC_solver.auto_CGD_for_loop(selectedImg_list= img_list, NA_range=[0,1],
    #                                         _lambda= _lambda)
    img_list = [0, 1, 2]
    initial_est = _PPR_DPC_solver.PPR_DPC(selectedImg_list=img_list, n_iter=3, linear_n_iter=60, lr=None, NA_range=[0, 1],
                                          _lambda=300)

    # Cascade Dark PPR ==================================================
    # img_list    = [0,1,2]
    # initial_est = _PPR_DPC_solver.PPR_DPC(selectedImg_list= img_list, n_iter= 2, linear_n_iter= 50, lr= None, NA_range=[0,1],
    #                                         _lambda=100)

    # initial_est = read_complex_str_csv('dataset/V2_new_phase_target/3_patterns_bright_PPR_DPC_niter=100_lniter=100_lambda=300_csv_file.csv')
    img_list = [0, 1, 2, 3, 4, 5]
    initial_est = _PPR_DPC_solver.PPR_DPC(selectedImg_list=img_list, n_iter=2, linear_n_iter=30, lr=None, NA_range=[1, 1.3], initial_est=initial_est,
                                          _lambda=700)

    # img_list    = [0,1,2,3,4,5,6,7,8]
    # initial_est = _PPR_DPC_solver.PPR_DPC(selectedImg_list= img_list, n_iter= 3, linear_n_iter= 4, lr= None, NA_range=[1.3,1.6], initial_est= initial_est,
    #                                         _lambda=2300)

    # img_list    = [0,1,2,3,4,5,6,7,8,12,13,14]
    # initial_est = _PPR_DPC_solver.PPR_DPC(selectedImg_list= img_list, n_iter= 5, linear_n_iter= 3, lr= None, NA_range=[1.6,2], initial_est= initial_est,
    #                                         _lambda=2700)

    # initial_est = read_complex_str_csv('dataset/V2_new_phase_target/3_patterns_bright_PPR_DPC_niter=100_lniter=100_lambda=300_csv_file.csv')
    # img_list    = [0,1,2,3,4,5]
    # for _lambda in [1500,1600,1700,1800,1900,2000]:
    #     _PPR_DPC_solver.auto_CGD_for_loop(selectedImg_list= img_list, NA_range=[1,1.3], initial_est= initial_est,
    #                                         _lambda= _lambda)

    # initial_est = read_complex_str_csv('dataset/V2_new_phase_target/3_patterns_dark_[1,1.3]_PPR_DPC_niter=3_lniter=5_lambda=1500_csv_file.csv')
    # img_list    = [0,1,2,3,4,5,6,7,8]
    # initial_est = _PPR_DPC_solver.PPR_DPC(selectedImg_list= img_list, n_iter= 4, linear_n_iter= 3, lr= None, NA_range=[1,1.6], initial_est= initial_est,
    #                                         _lambda=2400)

    # initial_est = read_complex_str_csv('dataset/V2_new_phase_target/3_patterns_dark_[1,1.6]_PPR_DPC_niter=4_lniter=3_lambda=2400_csv_file.csv')
    # img_list    = [0,1,2,3,4,5,6,7,8,12,13,14]
    # initial_est = _PPR_DPC_solver.PPR_DPC(selectedImg_list= img_list, n_iter= 2, linear_n_iter= 4, lr= None, NA_range=[1,2], initial_est= initial_est,
    #                                         _lambda=3500)

    # Direct Dark PPR ==================================================
    # initial_est = read_complex_str_csv('dataset/V2_new_phase_target/3_patterns_bright_PPR_DPC_niter=100_lniter=100_lambda=300_csv_file.csv')
    # img_list    = [0,1,2,15,16,17]
    # for _lambda in [1000,1100,1200,1300,1400]:
    #     initial_est = _PPR_DPC_solver.PPR_DPC(selectedImg_list= img_list, n_iter= 100, linear_n_iter= 100, lr= None, NA_range=[1,2], initial_est= initial_est,
    #                                             _lambda= _lambda)
    # initial_est = _PPR_DPC_solver.PPR_DPC(selectedImg_list= img_list, n_iter= 3, linear_n_iter= 100, lr= None, NA_range=[1,2], initial_est= initial_est,
    #                                             _lambda= 1000)
