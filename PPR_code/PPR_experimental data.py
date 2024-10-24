from pyphaseretrieve import phaseretrieval
from pyphaseretrieve import algos
from pyphaseretrieve.linop import *
import skimage.io as skio
from scipy import io
import matplotlib.font_manager as fm
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
import re
import csv
import matplotlib.cm as cm
import matplotlib as mpl
import matplotlib.pyplot as plt
import json
import math
import cupy as cp
import numpy as np
import os

RESOLUTION256_250_PATH = "dataset/V4_new_phase_target/0.25NA_250ms_10X/custom_250ms_centre[-240,295]_camerasize256_resolution_image.tif"
RESOLUTION256_250_BACKGROUND_PATH = "dataset/V4_new_phase_target/background_0.25NA_250ms_10X/background_custom_250ms_centre[-240,295]_camerasize256_resolution_image.tif"
RESOLUTION256_250_BACKREMOVE_AVERAGE50_PATH = "dataset/V4_new_phase_target/0.25NA_250ms_10X/custom_250ms_centre[-240,295]_backgrounRemove_average50_camerasize256_resolution_image.tif"
RESOLUTION256_250_BACKREMOVE_MEDIAN50_PATH = "dataset/V4_new_phase_target/0.25NA_250ms_10X/custom_250ms_centre[-240,295]_backgrounRemove_median50_camerasize256_resolution_image.tif"
RESOLUTION256_250_BACKREMOVE_AVERAGE500_PATH = "dataset/V4_new_phase_target/0.25NA_250ms_10X/custom_250ms_centre[-240,295]_backgrounRemove_average500_camerasize256_resolution_image.tif"
FPM_RESOLUTION_256_PATH = "dataset/V4_new_phase_target/0.25NA_1000ms_10X/custom_FPM_centre[-240,295]_camerasize256_resolution_image.tif"

RESOLUTION128_250_PATH = "dataset/V4_new_phase_target/0.25NA_250ms_10X/custom_250ms_centre[-240,295]_camerasize128_resolution_image.tif"
RESOLUTION128_250_BACKGROUND_PATH = "dataset/V4_new_phase_target/background_0.25NA_250ms_10X/background_custom_250ms_centre[-240,295]_camerasize128_resolution_image.tif"
RESOLUTION128_250_BACKREMOVE_AVERAGE50_PATH = "dataset/V4_new_phase_target/0.25NA_250ms_10X/custom_250ms_centre[-240,295]_backgrounRemove_average50_camerasize128_resolution_image.tif"
RESOLUTION128_250_BACKREMOVE_MEDIAN50_PATH = "dataset/V4_new_phase_target/0.25NA_250ms_10X/custom_250ms_centre[-240,295]_backgrounRemove_median50_camerasize128_resolution_image.tif"
RESOLUTION128_250_BACKREMOVE_AVERAGE500_PATH = "dataset/V4_new_phase_target/0.25NA_250ms_10X/custom_250ms_centre[-240,295]_backgrounRemove_average500_camerasize128_resolution_image.tif"
FPM_RESOLUTION_128_PATH = "dataset/V4_new_phase_target/0.25NA_1000ms_10X/custom_FPM_centre[-240,295]_camerasize128_resolution_image.tif"

LED_POS_NA_FILE_PATH = (
    "dataset/V4_new_phase_target/led_array_pos_na_z65mm.json"
)
LED_PATTERNS_FILE_PATH = "dataset/V4_new_phase_target/rings_led_patterns.json"

ANGLE_CALI_PATH = (
    "calibration_angle/V4 calib/Diffuser FPM 512 data_output_selfCal.mat"
)


# ================================================== START Dataset class ====================================================
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
        self.led_dia_number = 19
        self.total_LED_num = 245
        # generate experimental setup
        self.experimentSetup()

    # Initialize experimental parameters
    def experimentSetup(self):
        # information about LED
        with open(LED_POS_NA_FILE_PATH, "r") as open_file:
            self.led_pos_na_dict = json.load(open_file)
        # LED index of selected image
        with open(LED_PATTERNS_FILE_PATH, "r") as open_file:
            self.led_patterns_dict = json.load(open_file)

        self.fourier_res = self.get_fourierResolution()
        self.pupil_mask = self.get_pupil_mask()
        self.led_total_na_dict = self.get_led_total_na_dict()
        self.led_na_dict = self.get_led_na_dict()
        self.total_shifts_pair = self.get_shiftsPairs()

    # ----------------------- Default setup -----------------------
    def get_fourierResolution(self) -> float:
        # compute the fourier resolution of system
        img_pixel_size = self.camera_pixel_Gsize / self.mag
        FoV = img_pixel_size * self.camera_size
        fourier_resolution = 1 / FoV

        if 2 * self.NA / self.wave_lambda > (1 / img_pixel_size) / 2:
            print("ALIASING HAPPEN!")
        return fourier_resolution

    def get_pupil_mask(self):
        # generate the uniform pupil
        pupil_radius = self.NA / self.wave_lambda / self.fourier_res

        camera_size_idx = np.linspace(
            -math.floor(self.camera_size / 2),
            math.ceil(self.camera_size / 2) - 1,
            self.camera_size,
        )
        temp_mask_h, temp_mask_v = np.meshgrid(
            camera_size_idx, camera_size_idx
        )

        pupil_mask, _ = cart2pol(temp_mask_h, temp_mask_v)
        pupil_mask[pupil_mask <= pupil_radius] = 1
        pupil_mask[pupil_mask >= pupil_radius] = 0
        pupil_mask = np.fft.fftshift(pupil_mask)
        return pupil_mask

    # ----------------------- Read LED pos and NA -------------------------
    def get_led_total_na_dict(self) -> dict:
        total_na_dict = {}
        for idx, (key, value) in enumerate(
            self.led_pos_na_dict["led_position_list_na"].items()
        ):
            total_na_dict[key] = np.sqrt(value[0] ** 2 + value[1] ** 2)
            if int(key) == self.total_LED_num:
                break
        return total_na_dict

    def get_led_na_dict(self) -> dict:
        led_na_dict = {}
        for idx, (key, value) in enumerate(
            self.led_pos_na_dict["led_position_list_na"].items()
        ):
            led_na_dict[key] = value
            if int(key) == self.total_LED_num:
                break
        return led_na_dict

    def get_na_calib_array(self, path) -> np.ndarray:
        # read the angle calibration from mat file from Matlab code
        cali_mat = io.loadmat(path)
        na_calib = cali_mat["metadata"][0][0][0][0][0][1]
        return na_calib

    # ----------------------- NA and reconstruction -----------------------
    def get_reconstruction_size(
        self, total_selected_led_idx_array: np.ndarray
    ) -> int:
        if total_selected_led_idx_array is not None:
            NA_illu = self.get_illumnationNA_by_selected_index(
                total_selected_led_idx_array=total_selected_led_idx_array
            )
        else:
            raise NameError("Incorrect Input")
        synthetic_NA = NA_illu + self.NA
        max_reconstruction_size = math.floor(
            2 * synthetic_NA / self.wave_lambda / self.fourier_res
        )
        reconstruction_size = np.maximum(
            self.camera_size, max_reconstruction_size
        )

        print("illumination NA: {:.2f}".format(NA_illu))
        print("Total NA: {:.2f}".format(synthetic_NA))
        return reconstruction_size

    def get_illumnationNA_by_selected_index(
        self, total_selected_led_idx_array: np.ndarray
    ):
        temp_na_list = []
        for item in total_selected_led_idx_array:
            temp_na_list.append(self.led_total_na_dict[str(item)])

        illuminationNA = np.max(temp_na_list)
        return illuminationNA

    # ----------------------- Total shifts and shifts pairs -----------------------
    def get_shiftsPairs(self):
        # important parameters. create the Fourier shifts array for the system.
        shifts_h_list = []
        shifts_v_list = []
        for led_idx in range(self.total_LED_num):
            shifts_h_list.append(
                self.led_na_dict[str(led_idx)][0]
                / self.wave_lambda
                / self.fourier_res
                + 8
            )
            shifts_v_list.append(
                self.led_na_dict[str(led_idx)][1]
                / self.wave_lambda
                / self.fourier_res
                - 8
            )

        shifts_h = np.array(shifts_h_list)
        shifts_v = np.array(shifts_v_list)
        shifts_pair = np.concatenate(
            [
                shifts_v.reshape(self.total_LED_num, 1),
                shifts_h.reshape(self.total_LED_num, 1),
            ],
            axis=1,
        )

        # angle calibration from Matlab
        na_calib = self.get_na_calib_array(path=ANGLE_CALI_PATH)
        shifts_pair_calib = na_calib / self.wave_lambda / self.fourier_res
        shifts_pair[0:25, :] = shifts_pair_calib[0:25, :]

        return shifts_pair

    # ----------------------- Measurement selection methods -----------------------
    # Pattern json: contant LED index array. EX: image_1: [1,2,3,4,5]
    # LED pos and NA json: contant NA and position info of LED. EX: 1:{pos:[0,0], NA:[0,0]}

    def selectImg_by_singleDAT_selectImgList(
        self,
        img_num_array,
        centre: list = [0, 0],
        full_FOV: bool = False,
        input_size: int = None,
    ) -> np.ndarray:
        # load the image based on the index selection
        if full_FOV:
            h_start = int(
                self.CAMERA_H_RES // 2 + centre[0] - self.camera_size // 2
            )
            v_start = int(
                self.CAMERA_V_RES // 2 - centre[1] - self.camera_size // 2
            )
            crop_size = self.camera_size
        else:
            if input_size is None:
                input_size = self.camera_size
            h_start = int(input_size // 2 + centre[0] - self.camera_size // 2)
            v_start = int(input_size // 2 - centre[1] - self.camera_size // 2)
            crop_size = self.camera_size

        print("image loading...")
        if self.camera_size == 256:
            img_bright = skio.imread(
                RESOLUTION256_250_BACKREMOVE_AVERAGE500_PATH
            )
            img_dark = skio.imread(
                RESOLUTION256_250_BACKREMOVE_AVERAGE500_PATH
            )

        elif self.camera_size == 128:
            img_bright = skio.imread(
                RESOLUTION128_250_BACKREMOVE_AVERAGE500_PATH
            )
            img_dark = skio.imread(
                RESOLUTION128_250_BACKREMOVE_AVERAGE500_PATH
            )
        else:
            raise NameError("no data")

        y = None
        for image_key in img_num_array:
            if image_key < 3:
                img = img_bright
                single_img = img[
                    image_key,
                    v_start: v_start + crop_size,
                    h_start: h_start + crop_size,
                ]
            else:
                shrink_value = 1
                img = img_dark / shrink_value
                background_fac = (250 / 1000) / shrink_value
                single_img = img[
                    image_key,
                    v_start: v_start + crop_size,
                    h_start: h_start + crop_size,
                ]

                single_img_999_value = np.percentile(single_img, 99.9)
                single_img[single_img > single_img_999_value] = (
                    single_img_999_value
                )
                single_img[single_img < 0] = 0

            plt.figure()
            plt.imshow(single_img, cmap=cm.Greys_r)
            plt.colorbar()
            plt.title(f"measurement {image_key}")
            plt.savefig(
                f"_measurement_or_background/measurement {image_key}.png"
            )
            plt.close()

            if y is None:
                y = np.uint64(single_img)
            else:
                y = np.concatenate([y, np.uint64(single_img)], axis=0)
        print("finish loading...")
        return y

    def selectImg_2_multiplexMask_totalSelectArray(
        self, img_num_array, led_map_fig_or_not=True
    ):
        # concatenate all multiplex mask from selected img index
        pattern = "0.25NA"

        total_selected_led_idx_list = []
        multiplex_mask_list = []
        for idx_num, img_idx in enumerate(img_num_array):
            multiplex_mask_list.append(
                self.single_selectImg2multiplex_array(
                    selectImg_list=self.led_patterns_dict[pattern][img_idx]
                )
            )
            total_selected_led_idx_list = (
                total_selected_led_idx_list
                + self.led_patterns_dict[pattern][img_idx]
            )

            if led_map_fig_or_not:
                x_pos = []
                y_pos = []

                for selected_led in self.led_patterns_dict[pattern][img_idx]:
                    x_pos.append(
                        self.led_pos_na_dict["led_position_list_cartesian"][
                            str(selected_led)
                        ][0]
                    )
                    y_pos.append(
                        self.led_pos_na_dict["led_position_list_cartesian"][
                            str(selected_led)
                        ][1]
                    )

                plt.figure()
                plt.scatter(x=x_pos, y=y_pos)
                plt.colorbar()
                plt.title(f"LED pattern {idx_num+1}")
                plt.xlim(-31, 31)
                plt.ylim(-31, 31)
                plt.savefig(f"led_pattern/LED pattern {idx_num+1}")
                plt.close()
        multiplex_mask = np.array(multiplex_mask_list)

        return multiplex_mask, total_selected_led_idx_list

    def single_selectImg2multiplex_array(
        self, selectImg_list: list
    ) -> np.ndarray:
        # convert LED index to multiplex mask for library
        multiplex_array = np.zeros((self.total_LED_num,))
        for idx in range(self.total_LED_num):
            if idx in np.array(selectImg_list):
                multiplex_array[idx] = 1
        return multiplex_array


# ================================================ END Dataset ===================================================
# ================================================================================================================


# ============ START functions ============
# =========================================
def cart2pol(x, y):
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x) / np.pi * 180
    phi = (phi + 360) % 360
    return (rho, phi)


def delete_file(dir_name):
    dir_name = dir_name
    for file_name in os.listdir(dir_name):
        os.remove(os.path.join(dir_name, file_name))


def read_complex_str_csv(csv_path_file):
    # read csv file which store complex number.
    img = None
    with open(csv_path_file, newline="") as csvfile:
        spamreader = csv.reader(csvfile, delimiter=",")
        for idx_row, img_row in enumerate(spamreader):
            if img is None:
                img = np.zeros((len(img_row), len(img_row))).astype(
                    np.complex128
                )
            for idx_col, img_element in enumerate(img_row):
                num_of_element = re.findall(
                    "[-]\d+[.]\d+|\d+[.]\d+", img_element
                )
                power_of_element = re.findall("e[+,-]\d+", img_element)

                if len(num_of_element) == 1:
                    img[idx_row, idx_col] = float(num_of_element[0]) * float(
                        "1" + power_of_element[0]
                    )
                else:
                    img[idx_row, idx_col] = float(num_of_element[0]) * float(
                        "1" + power_of_element[0]
                    ) + 1j * (
                        float(num_of_element[1])
                        * float("1" + power_of_element[1])
                    )
    return img


# ============ END functions ============
# =======================================


# ================================================ START solver ===================================================
# ================================================================================================================


class PPR_DPC_solver(object):
    def __init__(self, camera_size: int, gpu=True) -> None:
        self.camera_size = camera_size
        self.dataset = new_custom_dataSet(camera_size=camera_size)
        self.gpu = gpu

    def Real_DPC(
        self,
        selectedImg_list: list,
        centre: list = [0, 0],
        alpha=0,
        file_pattern="2 halves",
        initial_est: np.ndarray = None,
        for_loop_or_not: bool = False,
        NA_range: list = [0, 1],
    ):
        # obtain reconstruction resolution base on LED in selected pattern.
        multiplex_led_array_mask, total_selected_led_idx_array = (
            self.dataset.selectImg_2_multiplexMask_totalSelectArray(
                img_num_array=selectedImg_list
            )
        )
        reconstruction_res = self.dataset.get_reconstruction_size(
            total_selected_led_idx_array=total_selected_led_idx_array
        )
        reconstruction_shape = (reconstruction_res, reconstruction_res)
        print(f"Real DPC recontruction shape: {reconstruction_shape}")

        # read pupil, shifts in F-space, intensity measurements
        DPC_y = np.array(
            self.dataset.selectImg_by_singleDAT_selectImgList(
                img_num_array=selectedImg_list, centre=centre
            )
        )
        probe = np.array(self.dataset.pupil_mask)
        total_shifts_pair = self.dataset.total_shifts_pair

        total_image = int(DPC_y.shape[0] / self.camera_size)
        DPC_y_list = []
        for idx_img in range(total_image):
            DPC_y_list.append(
                DPC_y[
                    idx_img * self.camera_size: (idx_img + 1)
                    * self.camera_size,
                    :,
                ]
            )

        # DPC intensity normalization
        for idx, img in enumerate(DPC_y_list):
            meanIntensity = np.mean(img)
            img = img / meanIntensity
            img = img - 1
            DPC_y_list[idx] = img

        # upsampling pupil to fit the dimension of reconstruction
        croplinop = LinOpCrop2(
            in_shape=(reconstruction_shape),
            crop_shape=(self.camera_size, self.camera_size),
        )
        probe = np.fft.ifftshift(croplinop.applyT(np.fft.fftshift(probe)))

        # compute phase transfer function
        transfer_func_list = []
        transfer_func_led_num = []
        for i_mask in multiplex_led_array_mask:
            transfer_func = np.zeros_like(probe)
            led_num = 0
            for idx, mask_item in enumerate(i_mask):
                if (mask_item != False) and (mask_item != 0):
                    pos_shift = LinOpRoll2(
                        total_shifts_pair[idx, 0], total_shifts_pair[idx, 1]
                    )
                    neg_shift = LinOpRoll2(
                        -total_shifts_pair[idx, 0], -total_shifts_pair[idx, 1]
                    )

                    if transfer_func is None:
                        transfer_func = np.copy(
                            neg_shift.apply(probe) - pos_shift.apply(probe)
                        )
                    else:
                        transfer_func = (
                            transfer_func
                            + neg_shift.apply(probe)
                            - pos_shift.apply(probe)
                        )
                    led_num = led_num + 1
            transfer_func = transfer_func / led_num
            transfer_func_list.append(transfer_func)

        # name of output images
        file_path = str(f"_recon_img/")
        file_header = str("DPC_transfor")
        file_pattern = str("pattern: " + file_pattern)
        file_end = str(".png")

        # solve the phase DPC
        sum_transfer_DPC_phase = np.zeros_like(probe)
        sum_transfer_func_square = np.zeros_like(probe)
        for idx, _transfer in enumerate(transfer_func_list):
            FT_DPC_y = np.fft.fft2(DPC_y_list[idx], norm="ortho")
            FT_DPC_y = (
                np.fft.ifftshift(croplinop.applyT(np.fft.fftshift(FT_DPC_y)))
                * reconstruction_res
                / self.camera_size
            )

            sum_transfer_DPC_phase = (
                sum_transfer_DPC_phase + (_transfer * 1j).conj() * FT_DPC_y
            )
            sum_transfer_func_square = (
                sum_transfer_func_square + np.abs(_transfer) ** 2
            )

            plt.figure()
            plt.imshow(np.fft.fftshift(_transfer), cmap=cm.Greys_r)
            plt.colorbar()
            plt.title(file_header + f" function{idx}, " + file_pattern)
            plt.savefig(
                file_path
                + file_header
                + f" function {idx} "
                + file_pattern
                + file_end
            )
            plt.close()

        FT_phase = sum_transfer_DPC_phase / (sum_transfer_func_square + alpha)
        x_est_phase = np.fft.ifft2(FT_phase, norm="ortho")

        # =========== START output image ===========
        x_est_phase = np.rot90(x_est_phase, -1)

        file_alpha = str(f"alpha={alpha}")
        plt.figure()
        plt.imshow(np.real(x_est_phase), cmap=cm.Greys_r)
        plt.colorbar()
        plt.title(file_header + f" phase, " + file_pattern)
        plt.savefig(
            file_path
            + file_header
            + " phase, "
            + file_alpha
            + f", "
            + file_pattern
            + file_end
        )
        plt.close()

        croplinop = LinOpCrop2(
            in_shape=(self.camera_size * 2, self.camera_size * 2),
            crop_shape=reconstruction_shape,
        )
        padded_phase_FT = croplinop.applyT(
            np.fft.fftshift(np.fft.fft2(x_est_phase, norm="ortho"))
        )
        padded_phase_img = (
            np.fft.ifft2(np.fft.ifftshift(padded_phase_FT), norm="ortho")
            * (self.camera_size * 2)
            / reconstruction_shape[0]
        )
        plt.figure()
        plt.imshow(np.real(padded_phase_img), cmap=cm.Greys_r)
        plt.colorbar()
        plt.title(f"Phase padding: " + file_pattern)
        plt.savefig(
            file_path
            + file_header
            + " Phase: padding 256, "
            + file_alpha
            + file_pattern
            + file_end
        )
        plt.close()

        phase_percent_1 = np.percentile(np.real(padded_phase_img), 99.9)
        pad_phase_recon = np.copy(np.real(padded_phase_img))
        pad_phase_recon[pad_phase_recon > phase_percent_1] = phase_percent_1
        plt.figure()
        plt.imshow(pad_phase_recon, cmap=cm.Greys_r)
        plt.colorbar()
        plt.title(f"Phase filter 0.1% : " + file_pattern)
        plt.savefig(
            file_path
            + file_header
            + " Phase: filter 0.1% "
            + file_alpha
            + file_pattern
            + file_end
        )
        plt.close()

        fig, ax = plt.subplots()
        cmap = cm.Greys_r
        norm = mpl.colors.Normalize(
            vmin=np.min(pad_phase_recon), vmax=np.max(pad_phase_recon)
        )
        fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax)

        ax.imshow(pad_phase_recon, cmap=cm.Greys_r)
        ax.set_title("Phase filter 0.1% with scalebar")
        fontprops = fm.FontProperties(size=12, weight="bold")
        scalebar = AnchoredSizeBar(
            ax.transData,
            62,
            "20 um",
            "lower center",
            pad=0.1,
            color="white",
            frameon=False,
            size_vertical=1,
            fontproperties=fontprops,
        )

        ax.add_artist(scalebar)
        plt.savefig(
            file_path
            + file_header
            + " Phase: filter 0.1% with scale bar "
            + file_alpha
            + file_pattern
            + file_end
        )
        plt.close()

        img = np.abs(np.fft.fftshift(np.fft.fft2(x_est_phase, norm="ortho")))
        img = croplinop.applyT(img) + 0.00001
        img = np.log(img)
        NA_radius = (
            self.dataset.NA
            / self.dataset.wave_lambda
            / self.dataset.fourier_res
        )
        _, axes = plt.subplots()
        plt_circle = plt.Circle(
            (self.camera_size, self.camera_size),
            NA_radius,
            fill=False,
            color="r",
        )
        axes.set_aspect(1)
        axes.add_artist(plt_circle)
        plt.imshow(img, cmap=cm.Greys_r)
        plt.colorbar()
        axes.set_title("Log Abs of FT Phase image " + file_pattern)
        plt.savefig(
            file_path
            + file_header
            + "Phase: Log FT image "
            + file_alpha
            + file_pattern
            + file_end
        )
        plt.close()

        plot_x = 321
        plot_y = [190, 233]
        y_width = (plot_y[1] - plot_y[0]) // 2
        plt.figure()
        plt.imshow(np.real(padded_phase_img), cmap=cm.Greys_r)
        plt.plot([plot_x, plot_x], [plot_y[0], plot_y[1]], "r-", lw=1.5)
        plt.colorbar()
        plt.title(f"Phase with line: " + file_pattern)
        plt.savefig(
            file_path
            + file_header
            + " Phase: with line "
            + file_alpha
            + file_pattern
            + file_end
        )
        plt.close()

        plt.figure()
        plt.imshow(
            np.real(
                padded_phase_img[
                    plot_y[0]: plot_y[1], plot_x - y_width: plot_x + y_width
                ]
            ),
            cmap=cm.Greys_r,
        )
        plt.plot([y_width, y_width], [0, 2 * y_width - 1], "r-", lw=1.5)
        plt.colorbar()
        plt.title(f"Phase: Zoom in " + file_pattern)
        plt.savefig(
            file_path
            + file_header
            + " Phase: Zoom in, "
            + file_alpha
            + file_pattern
            + file_end
        )
        plt.close()

        fig, ax = plt.subplots()
        cmap = cm.Greys_r
        norm = mpl.colors.Normalize(
            vmin=np.min(pad_phase_recon), vmax=np.max(pad_phase_recon)
        )
        fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax)
        ax.imshow(np.real(pad_phase_recon[140:290, 200:350]), cmap=cm.Greys_r)
        ax.set_title(f"Phase: Inset Zoom in " + file_pattern)
        plt.savefig(
            file_path
            + file_header
            + " Phase: Inset Zoom in, "
            + file_alpha
            + file_pattern
            + file_end
        )
        plt.close()

        y = np.real(padded_phase_img[plot_y[0]: plot_y[1], plot_x])
        x = np.linspace(0, y.shape[0], y.shape[0])
        plt.figure()
        plt.plot(x, y, "r-", lw=1.5)
        plt.title(f"Profile of cross-section: " + file_pattern)
        plt.savefig(
            file_path
            + file_header
            + " Profile of cross-section, "
            + file_alpha
            + file_pattern
            + file_end
        )
        plt.close()
        # =========== END output image ===========

    def PPR_DPC(
        self,
        selectedImg_list: list,
        n_iter=1,
        linear_n_iter=5,
        lr=1,
        centre: list = [0, 0],
        initial_est: np.ndarray = None,
        for_loop_or_not: bool = False,
        NA_range: list = [0, 1],
        _lambda=0,
        alpha=0,
    ):
        print("PPR start \n----------------------")

        # obtain reconstruction resolution base on LED in selected pattern.
        multiplex_led_array_mask, total_selected_led_idx_array = (
            self.dataset.selectImg_2_multiplexMask_totalSelectArray(
                img_num_array=selectedImg_list
            )
        )
        reconstruction_res = self.dataset.get_reconstruction_size(
            total_selected_led_idx_array=total_selected_led_idx_array
        )
        reconstruction_shape = (reconstruction_res, reconstruction_res)
        print(f"Recontruction shape: {reconstruction_shape}")

        # obtain pupil, shfit in F-space, measurments and forward model
        if self.gpu:
            y = cp.array(
                self.dataset.selectImg_by_singleDAT_selectImgList(
                    img_num_array=selectedImg_list, centre=centre
                )
            )
            probe = cp.array(self.dataset.pupil_mask)
        else:
            y = np.array(
                self.dataset.selectImg_by_singleDAT_selectImgList(
                    img_num_array=selectedImg_list, centre=centre
                )
            )
            probe = np.array(self.dataset.pupil_mask)
        total_shifts_pair = self.dataset.total_shifts_pair
        pr_model = phaseretrieval.MultiplexedPhaseRetrieval(
            probe=probe,
            multiplex_led_mask=multiplex_led_array_mask,
            shifts_pair=total_shifts_pair,
            reconstruct_shape=reconstruction_shape,
        )

        # initial guess
        if self.gpu:
            if initial_est is None:
                initial_est = cp.ones(
                    shape=reconstruction_shape, dtype=np.complex128
                )
            else:
                initial_est = cp.array(initial_est)
        else:
            if initial_est is None:
                initial_est = np.ones(
                    shape=reconstruction_shape, dtype=np.complex128
                )
            else:
                initial_est = np.array(initial_est)
        initial_est = np.fft.fft2(initial_est, norm="ortho")

        # intensity normalizztion of initial guess
        y_est = pr_model.apply_ModularSquare(initial_est)
        mean_y_est = np.mean(y_est[0: self.camera_size, :])
        mean_y = np.mean(y[0: self.camera_size, :])
        normalized_facter = np.sqrt(mean_y / mean_y_est)
        initial_est = initial_est * normalized_facter
        print(f"normalize factor: {normalized_facter}")

        # upsampling initial guess in case the dimensional mismatch
        crop_op = LinOpCrop2(
            in_shape=reconstruction_shape, crop_shape=initial_est.shape
        )
        initial_est = np.fft.ifftshift(
            crop_op.applyT(np.fft.fftshift(initial_est))
        )

        # algorithm solver
        ppr_method = algos.PerturbativePhase(pr_model)
        if lr is not None:
            x_est = ppr_method.iterate_GradientDescent(
                y=y,
                initial_est=initial_est,
                n_iter=n_iter,
                linear_n_iter=linear_n_iter,
                lr=lr,
                alpha=alpha,
            )
        else:
            x_est = ppr_method.iterate_ConjugateGradientDescent(
                y=y,
                initial_est=initial_est,
                n_iter=n_iter,
                linear_n_iter=linear_n_iter,
                _lambda=_lambda,
            )

        x_est = np.fft.ifft2(x_est, norm="ortho")

        if self.gpu:
            x_est = cp.asnumpy(x_est)

        # =========== START output image ===========
        x_est = np.rot90(x_est, -1)

        if for_loop_or_not:
            if lr is None:
                file_path = str(f"_PPR_DPC/CGD/n_iter={n_iter}")
            else:
                file_path = str(f"_PPR_DPC/GD/n_iter={n_iter}")
        else:
            file_path = str(f"_recon_img")
        file_header = str("/ResolutionImage_PPR-DPC_")
        if lr is None:
            file_iter = str(
                f"n_iter={n_iter}, l_n_iter={linear_n_iter}, lambda = {
                    _lambda}, NA_range[{NA_range[0]}, {NA_range[1]}]"
            )
        else:
            file_iter = str(
                f"n_iter={n_iter}, l_n_iter={linear_n_iter}, lr={
                    lr}, NA_range[{NA_range[0]}, {NA_range[1]}]"
            )
        file_end = str(".png")

        plt.figure()
        plt.imshow(np.abs(x_est) ** 2, cmap=cm.Greys_r)
        plt.colorbar()
        plt.title(f"Intensity: " + file_iter)
        plt.savefig(
            file_path + file_header + "Intensity: " + file_iter + file_end
        )
        plt.close()

        plt.figure()
        plt.imshow(np.angle(x_est), cmap=cm.Greys_r)
        plt.colorbar()
        plt.title(f"Phase: " + file_iter)
        plt.savefig(file_path + file_header + "Phase: " + file_iter + file_end)
        plt.close()

        croplinop = LinOpCrop2(
            in_shape=(self.camera_size * 2, self.camera_size * 2),
            crop_shape=reconstruction_shape,
        )
        img = np.abs(
            (np.fft.fftshift(np.fft.fft2(np.angle(x_est), norm="ortho")))
        )
        img = croplinop.applyT(img) + 0.0001
        img = np.log(img)

        NA_radius = (
            self.dataset.NA
            / self.dataset.wave_lambda
            / self.dataset.fourier_res
        )

        _, axes = plt.subplots()
        plt_circle = plt.Circle(
            (self.camera_size, self.camera_size),
            NA_radius,
            fill=False,
            color="r",
        )
        axes.set_aspect(1)
        axes.add_artist(plt_circle)
        plt.imshow(img, cmap=cm.Greys_r)
        plt.colorbar()
        axes.set_title("Log Abs of FT Phase image " + file_iter)
        plt.savefig(
            file_path
            + file_header
            + "Phase: Log FT image "
            + file_iter
            + file_end
        )
        plt.close()

        padded_phase_FT = croplinop.applyT(
            np.fft.fftshift(np.fft.fft2(x_est, norm="ortho"))
        )
        padded_phase_img = np.angle(
            np.fft.ifft2(np.fft.ifftshift(padded_phase_FT), norm="ortho")
            * (self.camera_size * 2)
            / reconstruction_shape[0]
        )
        plt.figure()
        plt.imshow(padded_phase_img, cmap=cm.Greys_r)
        plt.colorbar()
        plt.title(f"Phase: " + file_iter)
        plt.savefig(
            file_path
            + file_header
            + "Phase: padding 256: "
            + file_iter
            + file_end
        )
        plt.close()

        phase_percent_1 = np.percentile(padded_phase_img, 99.9)
        pad_phase_recon = np.copy(padded_phase_img)
        pad_phase_recon[pad_phase_recon > phase_percent_1] = phase_percent_1
        plt.figure()
        plt.imshow(pad_phase_recon, cmap=cm.Greys_r)
        plt.colorbar()
        plt.title(f"Phase filter 0.1% : " + file_iter)
        plt.savefig(
            file_path
            + file_header
            + "Phase: filter 0.1% "
            + file_iter
            + file_end
        )
        plt.close()

        plot_x = 321
        plot_y = [190, 233]  # [213, 233]
        y_width = (plot_y[1] - plot_y[0]) // 2
        plt.figure()
        plt.imshow(np.real(padded_phase_img), cmap=cm.Greys_r)
        plt.plot([plot_x, plot_x], [plot_y[0], plot_y[1]], "r-", lw=1.5)
        plt.colorbar()
        plt.title(f"Phase with line: " + file_iter)
        plt.savefig(
            file_path
            + file_header
            + " Phase: with line "
            + file_iter
            + file_end
        )
        plt.close()

        plt.figure()
        plt.imshow(
            np.real(
                padded_phase_img[
                    plot_y[0]: plot_y[1], plot_x - y_width: plot_x + y_width
                ]
            ),
            cmap=cm.Greys_r,
        )
        plt.plot([y_width, y_width], [0, 2 * y_width - 1], "r-", lw=1.5)
        plt.colorbar()
        plt.title(f"Phase: Zoom in " + file_iter)
        plt.savefig(
            file_path
            + file_header
            + " Phase: Zoom in, "
            + file_iter
            + file_end
        )
        plt.close()

        fig, ax = plt.subplots()
        cmap = cm.Greys_r
        norm = mpl.colors.Normalize(
            vmin=np.min(pad_phase_recon), vmax=np.max(pad_phase_recon)
        )
        fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax)
        ax.imshow(np.real(pad_phase_recon[140:290, 200:350]), cmap=cm.Greys_r)
        ax.set_title(f"Phase: Inset Zoom in " + file_iter)
        plt.savefig(
            file_path
            + file_header
            + " Phase: Inset Zoom in, "
            + file_iter
            + file_end
        )
        plt.close()

        y = np.real(padded_phase_img[plot_y[0]: plot_y[1], plot_x])
        x = np.linspace(0, y.shape[0], y.shape[0])
        plt.figure()
        plt.plot(x, y, "r-", lw=1.5)
        plt.title(f"Profile of cross-section: " + file_iter)
        plt.savefig(
            file_path
            + file_header
            + " Profile of cross-section, "
            + file_iter
            + file_end
        )
        plt.close()

        x_est = np.rot90(x_est, 1)
        # =========== END output image ===========

        return x_est


# ================================================ END solver ===================================================
# ================================================================================================================


if __name__ == "__main__":
    # clean folder
    delete_file("led_pattern")
    delete_file("_recon_img")
    delete_file("_measurement_or_background")

    _PPR_DPC_solver = PPR_DPC_solver(camera_size=256)
    centre = [0, 0]

    # Real DPC ========================================================
    img_list = [0, 1]
    _PPR_DPC_solver.Real_DPC(
        selectedImg_list=img_list, alpha=0.1, centre=centre
    )

    # Bright PPR ======================================================
    img_list = [0, 1, 2]
    initial_est = _PPR_DPC_solver.PPR_DPC(
        selectedImg_list=img_list,
        n_iter=3,
        linear_n_iter=6,
        lr=None,
        NA_range=[0, 1],
        centre=centre,
        _lambda=0,
    )
    # Direct Dark PPR ==================================================
    img_list = [0, 1, 2, 8]
    _PPR_DPC_solver.PPR_DPC(
        selectedImg_list=img_list,
        n_iter=4,
        linear_n_iter=5,
        lr=None,
        NA_range=[1, 2],
        initial_est=initial_est,
        centre=centre,
        _lambda=0,
    )

    img_list = [0, 1, 2, 6, 7]
    _PPR_DPC_solver.PPR_DPC(
        selectedImg_list=img_list,
        n_iter=4,
        linear_n_iter=5,
        lr=None,
        NA_range=[1, 1.5],
        initial_est=initial_est,
        centre=centre,
        _lambda=0,
    )

    img_list = [0, 1, 2, 3, 4, 5]
    _PPR_DPC_solver.PPR_DPC(
        selectedImg_list=img_list,
        n_iter=6,
        linear_n_iter=4,
        lr=None,
        NA_range=[1.3, 1.7],
        initial_est=initial_est,
        centre=centre,
        _lambda=0,
    )
