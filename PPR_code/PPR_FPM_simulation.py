from typing import Optional
import numpy as np
import torch as th

import skimage

import pyphaseretrieve.linop as pl
from pyphaseretrieve import algos
from pyphaseretrieve import phaseretrieval
from pyphaseretrieve import loss


device = th.device("cpu")


class U2OS_cell_dataSet(object):
    def __init__(self, camera_size: int) -> None:
        ## Experiment Setup Parameters Setting (distance and length unit: um)
        # Optical system
        self.wave_lambda = 0.514
        self.NA = 0.19
        # Camera system
        self.camera_size = camera_size
        self.mag = 8.1485
        self.camera_pixel_Gsize = 5.5
        self.CAMERA_H_RES = 512
        self.CAMERA_V_RES = 512
        # LED system
        self.led_pitch = 4000
        self.led_d_z = 67500
        self.led_dia_number = 25
        self.total_n_img = 293
        ## generate experimental setup
        self.experimentSetup()

    # Initialize experimental parameters
    def experimentSetup(self):
        self.fourier_res = self.get_fourierResolution()
        self.pupil_mask = self.get_pupil_mask()
        (
            self.total_shifts_h_map,
            self.total_shifts_v_map,
            self.total_shifts_pair,
            *_,
        ) = self.get_shiftsMap_shiftsPairs()

    def get_fourierResolution(self) -> float:
        img_pixel_size = self.camera_pixel_Gsize / self.mag
        FoV = img_pixel_size * self.camera_size
        fourier_resolution = 1 / FoV

        if 2 * self.NA / self.wave_lambda > (1 / img_pixel_size) / 2:
            print("There is aliasing")
        return fourier_resolution

    def get_pupil_mask(self):
        pupil_radius = self.NA / self.wave_lambda / self.fourier_res

        camera_size_idx = th.linspace(
            -np.floor(self.camera_size / 2),
            np.ceil(self.camera_size / 2) - 1,
            self.camera_size,
            device=device,
        )
        temp_mask_h, temp_mask_v = th.meshgrid(
            camera_size_idx, camera_size_idx, indexing="ij"
        )

        pupil_mask, _ = cart2pol(temp_mask_h, temp_mask_v)
        pupil_mask[pupil_mask <= pupil_radius] = 1
        pupil_mask[pupil_mask >= pupil_radius] = 0
        return th.fft.fftshift(pupil_mask)

    def get_reconstruction_size(
        self,
        bright_field_NA: bool = False,
        multiplex_led_array_mask: th.Tensor = None,
        select_led_index_array: th.Tensor = None,
    ) -> int:
        if bright_field_NA:
            NA_illu = self.get_bright_illumnationNA()
        elif multiplex_led_array_mask is not None:
            NA_illu = self.get_illumnationNA_by_mulitplex_array(
                multiplex_led_array_mask=multiplex_led_array_mask
            )
        elif select_led_index_array is not None:
            NA_illu = self.get_illumnationNA_by_index(
                select_led_index_array=select_led_index_array
            )
        else:
            NA_illu = self.get_total_illumnationNA()
        synthetic_NA = NA_illu + self.NA
        print("illumination NA: {:.3f}".format(NA_illu))
        print("Total NA: {:.3f}".format(synthetic_NA))
        max_reconstruction_size = int(
            np.ceil(2 * synthetic_NA / self.wave_lambda / self.fourier_res)
        )
        reconstruction_size = np.maximum(
            int(self.camera_size), int(max_reconstruction_size)
        )
        return reconstruction_size

    def get_total_illumnationNA(self) -> float:
        led_r_number = np.floor(self.led_dia_number / 2)
        led_r = led_r_number * self.led_pitch
        illuminationNA = led_r / np.sqrt(self.led_d_z**2 + led_r**2)
        return illuminationNA

    def get_bright_illumnationNA(self) -> float:
        led_r_map, _ = self.get_led_r_angle_map()
        _, bright_field_led_mask, _ = self.get_bright_field_LED_map()
        bright_field_led_radius_map = led_r_map * bright_field_led_mask

        led_r = int(th.max(bright_field_led_radius_map)) * self.led_pitch
        illuminationNA = led_r / np.sqrt(self.led_d_z**2 + led_r**2)
        return illuminationNA

    def get_illumnationNA_by_mulitplex_array(
        self, multiplex_led_array_mask: th.Tensor
    ) -> float:
        led_r_map, _ = self.get_led_r_angle_map()
        led_bool_mask = self.get_led_bool_mask()
        led_r_array = led_r_map[led_bool_mask]

        illuminationNA = 0
        for i_multiplex_array in multiplex_led_array_mask:
            temp_led_r = (
                int(th.max(led_r_array[(i_multiplex_array > 0)]))
                * self.led_pitch
            )
            temp_illNA = temp_led_r / np.sqrt(self.led_d_z**2 + temp_led_r**2)
            if temp_illNA > illuminationNA:
                illuminationNA = temp_illNA
        return illuminationNA

    def get_illumnationNA_by_index(self, select_led_index_array: th.Tensor):
        led_index_map = self.get_led_index_map()
        led_radius_map, _ = self.get_led_r_angle_map()

        radius_list = []
        for idx_item in select_led_index_array:
            index = th.where(led_index_map == idx_item)
            radius_list.append(led_radius_map[index])

        led_r = int(th.max(radius_list)) * self.led_pitch
        illuminationNA = led_r / np.sqrt(self.led_d_z**2 + led_r**2)
        return illuminationNA

    def get_led_bool_mask(self) -> th.Tensor:
        led_ra_size = np.floor(self.led_dia_number / 2)
        led_h_idx_array = th.linspace(
            -led_ra_size, led_ra_size, self.led_dia_number
        )
        led_v_idx_array = th.linspace(
            -led_ra_size, led_ra_size, self.led_dia_number
        )

        led_h_idx_map, led_v_idx_map = th.meshgrid(
            led_h_idx_array, led_v_idx_array, indexing="ij"
        )
        led_r_map, _ = cart2pol(led_h_idx_map, led_v_idx_map)
        led_bool_mask = led_r_map < self.led_dia_number / 2
        return led_bool_mask

    def get_led_index_map(self) -> th.Tensor:
        led_bool_mask = self.get_led_bool_mask()
        led_index_map = th.ones((self.led_dia_number, self.led_dia_number))
        led_index_map = led_index_map * led_bool_mask
        led_index_map = led_index_map.reshape(
            self.led_dia_number**2,
        )
        led_idx = 1
        for idx in range(self.led_dia_number**2):
            if led_index_map[idx] != 0:
                led_index_map[idx] = led_idx
                led_idx += 1
        led_index_map = led_index_map.reshape(
            self.led_dia_number, self.led_dia_number
        )
        return led_index_map

    def get_led_r_angle_map(self) -> th.Tensor:
        led_ra_size = np.floor(self.led_dia_number / 2)
        led_h_idx_array = th.linspace(
            -led_ra_size, led_ra_size, self.led_dia_number
        )
        led_v_idx_array = th.linspace(
            -led_ra_size, led_ra_size, self.led_dia_number
        )

        led_h_idx_map, led_v_idx_map = th.meshgrid(
            led_h_idx_array, led_v_idx_array, indexing="ij"
        )
        led_r_map, led_angle_map = cart2pol(led_h_idx_map, led_v_idx_map)
        return led_r_map, led_angle_map

    def get_shiftsMap_shiftsPairs(self):
        led_r_map, _ = self.get_led_r_angle_map()
        led_r_map[led_r_map < self.led_dia_number / 2] = 1
        led_r_map[led_r_map > self.led_dia_number / 2] = 0

        led_ra_size = np.floor(self.led_dia_number / 2)
        led_h_idx_array = th.linspace(
            -led_ra_size, led_ra_size, self.led_dia_number
        )
        led_v_idx_array = th.linspace(
            -led_ra_size, led_ra_size, self.led_dia_number
        )
        led_h_idx_map, led_v_idx_map = th.meshgrid(
            led_h_idx_array, led_v_idx_array, indexing="ij"
        )
        led_h_d_map = led_h_idx_map * self.led_pitch
        led_v_d_map = led_v_idx_map * self.led_pitch
        led_to_sample_dist_map = th.sqrt(
            (led_h_d_map) ** 2 + (led_v_d_map) ** 2 + self.led_d_z**2
        )

        sinTheta_h_map = led_h_d_map / led_to_sample_dist_map
        sinTheta_v_map = led_v_d_map / led_to_sample_dist_map

        shifts_h_map = th.round(
            sinTheta_h_map * led_r_map / self.wave_lambda / self.fourier_res
        )
        shifts_v_map = th.round(
            sinTheta_v_map * led_r_map / self.wave_lambda / self.fourier_res
        )

        n_used_led = int(th.sum(led_r_map))
        shifts_h = shifts_h_map[led_r_map == 1]
        shifts_v = shifts_v_map[led_r_map == 1]
        shifts_pair = th.concatenate(
            [
                shifts_v.reshape(n_used_led, 1),
                shifts_h.reshape(n_used_led, 1),
            ],
            axis=1,
        )

        return (
            shifts_h_map,
            shifts_v_map,
            shifts_pair,
            sinTheta_h_map,
            sinTheta_v_map,
            led_r_map,
        )

    def get_bright_field_LED_map(self):
        _, _, _, sinTheta_h_map, sinTheta_v_map, _ = (
            self.get_shiftsMap_shiftsPairs()
        )
        led_NA_map = th.sqrt(sinTheta_h_map**2 + sinTheta_v_map**2)

        led_index_map = self.get_led_index_map()

        bright_field_led_bool_mask = led_NA_map <= self.NA
        bright_field_led_mask = bright_field_led_bool_mask.to(th.int32)
        bright_field_led_index_map = (
            bright_field_led_mask * led_index_map
        ).to(th.int32)
        return (
            bright_field_led_index_map,
            bright_field_led_mask,
            bright_field_led_bool_mask,
        )

    def get_bright_field_multiplex_led_array_mask(
        self, angle_range: th.Tensor
    ) -> th.Tensor:
        led_r_map, led_angle_map = self.get_led_r_angle_map()
        led_index_map = self.get_led_index_map()
        print(led_index_map)
        led_bool_mask = self.get_led_bool_mask()
        _, bright_field_led_mask, _ = self.get_bright_field_LED_map()

        multiplex_led_array_mask_list = []
        for idx in range(angle_range.shape[0]):
            if angle_range[idx, 0] is None:
                led_angle_bool_mask = led_r_map == 0
            else:
                if (angle_range[idx, 1] - angle_range[idx, 0]) < 0:
                    led_angle_bool_mask = th.logical_and(
                        th.logical_or(
                            angle_range[idx, 0] < led_angle_map,
                            led_angle_map < angle_range[idx, 1],
                        ),
                        led_r_map != 0,
                    )
                else:
                    led_angle_bool_mask = th.logical_and(
                        angle_range[idx, 0] < led_angle_map,
                        led_angle_map < angle_range[idx, 1],
                    )
                    if angle_range[idx, 1] > 360:
                        led_angle_bool_mask = th.logical_and(
                            angle_range[idx, 0] <= led_angle_map,
                            led_angle_map < angle_range[idx, 1],
                        )

            multiplex_led_idx_map = (
                led_index_map
                * led_bool_mask
                * bright_field_led_mask
                * led_angle_bool_mask
            )


            multiplex_led_bool_array_mask = (
                multiplex_led_idx_map[led_bool_mask] > 1
            )

            import matplotlib.pyplot as plt
            plt.figure()
            plt.imshow(led_bool_mask)
            plt.show()
            print(multiplex_led_idx_map)
            multiplex_led_array_mask = multiplex_led_bool_array_mask.to(
                th.int32
            )
            multiplex_led_array_mask_list.append(multiplex_led_array_mask)

        multiplex_led_array_mask = th.empty(
            (
                len(multiplex_led_array_mask_list),
                *multiplex_led_array_mask_list[0].shape,
            ),
            device=device,
        )
        for i, mask in enumerate(multiplex_led_array_mask_list):
            multiplex_led_array_mask[i] = mask.to(device)

        return multiplex_led_array_mask

    def get_single_dark_multiplex_led_array_mask(
        self, single_angle_range: list, single_radius_range: list
    ) -> th.Tensor:
        # Convert led_angle_map to 0-360 degree map
        led_r_map, led_angle_map = self.get_led_r_angle_map()

        dark_field_radius_range_led_bool_mask = th.logical_and(
            single_radius_range[0] <= led_r_map,
            led_r_map <= single_radius_range[1],
        )
        _, _, bright_field_led_bool_mask = self.get_bright_field_LED_map()
        dark_field_radius_range_led_bool_mask[bright_field_led_bool_mask] = (
            False
        )

        if (single_angle_range[1] - single_angle_range[0]) < 0:
            dark_field_angle_range_led_bool_mask = th.logical_and(
                th.logical_or(
                    single_angle_range[0] < led_angle_map,
                    led_angle_map < single_angle_range[1],
                ),
                led_r_map != 0,
            )
        else:
            dark_field_angle_range_led_bool_mask = th.logical_and(
                single_angle_range[0] < led_angle_map,
                led_angle_map < single_angle_range[1],
            )
            if single_angle_range[1] > 360:
                dark_field_angle_range_led_bool_mask = th.logical_and(
                    single_angle_range[0] <= led_angle_map,
                    led_angle_map < single_angle_range[1],
                )

        led_index_map = self.get_led_index_map()
        led_bool_mask = self.get_led_bool_mask()
        multiplex_led_idx_map = (
            led_index_map
            * led_bool_mask
            * dark_field_radius_range_led_bool_mask
            * dark_field_angle_range_led_bool_mask
        )
        multiplex_led_idx_array = np.array(
            multiplex_led_idx_map[led_bool_mask]
        )
        multiplex_led_mask = (multiplex_led_idx_map > 0).to(th.int32)
        multiplex_led_array_mask = (multiplex_led_idx_array > 0).to(th.int32)

        return (
            multiplex_led_idx_map,
            multiplex_led_idx_array,
            multiplex_led_mask,
            multiplex_led_array_mask,
        )

    def get_dark_multiplex_led_array_mask(
        self,
        multi_angle_range: list,
        multi_radius_range: list,
    ) -> th.Tensor:
        multiplex_led_array_mask_list = []

        for idx_lists, angle_lists in enumerate(multi_angle_range):
            single_multiplex_led_array_mask = None

            if isinstance(angle_lists[0], list):
                multiplex_led_mask = None
                for idx_list, angle_list in enumerate(angle_lists):
                    _, _, _multiplex_led_mask, _multiplex_led_array_mask = (
                        self.get_single_dark_multiplex_led_array_mask(
                            single_angle_range=angle_list,
                            single_radius_range=multi_radius_range[idx_lists][
                                idx_list
                            ],
                        )
                    )

                    if single_multiplex_led_array_mask is None:
                        single_multiplex_led_array_mask = (
                            _multiplex_led_array_mask
                        )
                    else:
                        single_multiplex_led_array_mask += (
                            _multiplex_led_array_mask
                        )

                    if multiplex_led_mask is None:
                        multiplex_led_mask = _multiplex_led_mask
                    else:
                        multiplex_led_mask += _multiplex_led_mask

            else:
                _, _, _multiplex_led_mask, _multiplex_led_array_mask = (
                    self.get_single_dark_multiplex_led_array_mask(
                        single_angle_range=angle_lists,
                        single_radius_range=multi_radius_range[idx_lists],
                    )
                )
                single_multiplex_led_array_mask = _multiplex_led_array_mask
                multiplex_led_mask = _multiplex_led_mask

            multiplex_led_array_mask_list.append(
                single_multiplex_led_array_mask
            )

        multiplex_led_array_mask = np.array(multiplex_led_array_mask_list)
        return multiplex_led_array_mask

    def select_FPM_image_by_NA(self, radius: float) -> th.Tensor:
        led_r_map, _ = self.get_led_r_angle_map()
        led_index_map = self.get_led_index_map()

        select_led_bool_r_map = led_r_map <= radius
        select_led_index_map = led_index_map * select_led_bool_r_map
        select_led_index_array = select_led_index_map[
            select_led_index_map > 0
        ].to(th.int32)

        shifts_pair = self.total_shifts_pair[(select_led_index_array - 1), 0:2]

        return select_led_index_array, shifts_pair

    def select_bright_FPM_image(self) -> th.Tensor:
        bright_field_led_index_map, _, _ = self.get_bright_field_LED_map()

        bright_FPM_index_array = bright_field_led_index_map[
            bright_field_led_index_map > 0
        ]
        shifts_pair = self.total_shifts_pair[(bright_FPM_index_array - 1), 0:2]
        return bright_FPM_index_array, shifts_pair


def cart2pol(x, y):
    rho = th.sqrt(x**2 + y**2)
    phi = th.arctan2(y, x) / th.pi * 180
    phi = (phi + 360) % 360
    return (rho, phi)


class DPC_and_darkField_solver:
    def __init__(self, camera_size: int) -> None:
        self.camera_size = camera_size
        self.dataset = U2OS_cell_dataSet(camera_size=camera_size)

    def Real_DPC(
        self,
        image,
        bright_field_angle_range: th.Tensor,
        groundtruth_size: int,
        centre: list = [0, 0],
        alpha=0,
    ):
        # obtain reconstruction resolution base on LED in selected pattern.
        DPC_multiplex_led_array_mask = (
            self.dataset.get_bright_field_multiplex_led_array_mask(
                angle_range=bright_field_angle_range
            )
        )
        print(DPC_multiplex_led_array_mask)
        reconstruction_res = self.dataset.get_reconstruction_size(
            bright_field_NA=True
        )
        reconstruction_shape = (reconstruction_res, reconstruction_res)
        print(f"Real DPC recontruction shape: {reconstruction_shape}")

        total_shifts_pair = self.dataset.total_shifts_pair
        probe = self.dataset.get_pupil_mask()

        # obatin forward model and generate the measurements
        all_shifts = []
        for mask in DPC_multiplex_led_array_mask:
            angle_shifts = total_shifts_pair[mask.to(th.bool)]
            all_shifts.append(angle_shifts)
        pr_model = phaseretrieval.MultiplexedFourierPtychography(
            probe=probe,
            shifts=all_shifts,
            reconstruct_shape=reconstruction_shape,
        )
        DPC_y = pr_model.forward(th.fft.fft2(image, norm="ortho"))
        total_image = int(DPC_y.shape[0] / self.camera_size)

        # seperate the concatenated measurements
        DPC_y_list = []
        for idx_img in range(total_image):
            DPC_y_list.append(
                DPC_y[
                    idx_img * self.camera_size : (idx_img + 1)
                    * self.camera_size,
                    :,
                ]
            )

        # upsampling pupil to fit the dimension of reconstruction if needed
        croplinop = pl.LinOpCrop2(
            in_shape=(reconstruction_shape),
            crop_shape=(self.camera_size, self.camera_size),
        )
        probe = th.fft.ifftshift(croplinop.applyT(th.fft.fftshift(probe)))

        # compute phase transfer function
        transfer_func_list = []
        for i_mask in DPC_multiplex_led_array_mask:
            transfer_func = th.zeros_like(probe)
            for idx, mask_item in enumerate(i_mask):
                if (mask_item != False) and (mask_item != 0):
                    pos_shift = pl.LinOpRoll2(
                        total_shifts_pair[idx, 0], total_shifts_pair[idx, 1]
                    )
                    neg_shift = pl.LinOpRoll2(
                        -total_shifts_pair[idx, 0], -total_shifts_pair[idx, 1]
                    )

                    if transfer_func is None:
                        transfer_func = th.copy(
                            neg_shift.apply(probe) - pos_shift.apply(probe)
                        )
                    else:
                        transfer_func = (
                            transfer_func
                            + neg_shift.apply(probe)
                            - pos_shift.apply(probe)
                        )
            transfer_func_list.append(transfer_func)

        # solve the phase DPC
        sum_transfer_DPC_phase = th.zeros_like(probe)
        sum_transfer_func_square = th.zeros_like(probe)
        for idx, _transfer in enumerate(transfer_func_list):
            FT_DPC_y = th.fft.fft2(DPC_y_list[idx], norm="ortho")
            FT_DPC_y = (
                th.fft.ifftshift(croplinop.applyT(th.fft.fftshift(FT_DPC_y)))
                * reconstruction_res
                / self.camera_size
            )

            sum_transfer_DPC_phase = (
                sum_transfer_DPC_phase + (_transfer * 1j).conj() * FT_DPC_y
            )
            sum_transfer_func_square = (
                sum_transfer_func_square + th.abs(_transfer) ** 2
            )

        FT_phase = sum_transfer_DPC_phase / (sum_transfer_func_square + alpha)
        x_est_phase = th.fft.ifft2(FT_phase, norm="ortho")

        return x_est_phase

    def BF_PPR(
        self,
        bright_field_angle_range: th.Tensor,
        groundtruth_size: int,
        n_iter=1,
        linear_n_iter=1,
        lr=5,
        centre: list = [0, 0],
        alpha=0,
        _lambda=0,
    ):
        print("BF-PPR start \n----------------------")


        # obtain reconstruction resolution base on LED in selected pattern.
        self.DPC_multiplex_led_array_mask = (
            self.dataset.get_bright_field_multiplex_led_array_mask(
                angle_range=bright_field_angle_range
            )
        )
        reconstruction_res = self.dataset.get_reconstruction_size(
            bright_field_NA=True
        )
        reconstruction_shape = (
            reconstruction_res.item(),
            reconstruction_res.item(),
        )
        print(f"DPC recontruction shape: {reconstruction_shape}")

        # read pupil, shifts in F-space
        probe = self.dataset.get_pupil_mask()
        total_shifts_pair = self.dataset.total_shifts_pair

        # obatin forward model and generate the measurements
        pr_model = phaseretrieval.MultiplexedFourierPtychography(
            probe=probe,
            multiplex_led_mask=self.DPC_multiplex_led_array_mask,
            shifts_pair=total_shifts_pair,
            reconstruct_shape=reconstruction_shape,
        )
        # TODO why is that nor in the forward?
        DPC_y = pr_model.forward(th.fft.fft2(simu_img, norm="ortho"))

        # initial guess
        initial_est = th.ones(
            reconstruction_shape,
            dtype=th.complex64,
            device=device,
        )
        initial_est = th.fft.fft2(initial_est, norm="ortho")

        # algorithm solver
        ppr_method = algos.PerturbativePhase(pr_model)
        if lr is not None:
            x_est = ppr_method.iterate_GradientDescent(
                y=DPC_y,
                initial_est=initial_est,
                n_iter=n_iter,
                linear_n_iter=linear_n_iter,
                lr=lr,
                alpha=alpha,
            )
        else:
            x_est = ppr_method.iterate_ConjugateGradientDescent(
                y=DPC_y,
                x0=initial_est,
                n_iter=n_iter,
                linear_n_iter=linear_n_iter,
                tolerance=1e-20,
                _lambda=_lambda,
            )

        x_est = th.fft.ifft2(x_est, norm="ortho")
        return x_est

    def DF_PPR(
        self,
        image,
        bright_field_angle_range: th.Tensor,
        dark_multi_angle_range_list: list,
        dark_multi_radius_range_list: list,
        bright_n_iter=1,
        bright_linear_n_iter=5,
        bright_lr=None,
        dark_n_iter=15,
        dark_linear_n_iter=5,
        dark_lr=None,
        centre: list = [0, 0],
        NA_range=[1, 1],
    ):
        # generate multiplxed mask from BF and DF
        self.DPC_multiplex_led_array_mask = (
            self.dataset.get_bright_field_multiplex_led_array_mask(
                angle_range=bright_field_angle_range
            )
        )
        self.dark_multiplex_led_array_mask = (
            self.dataset.get_dark_multiplex_led_array_mask(
                multi_angle_range=dark_multi_angle_range_list,
                multi_radius_range=dark_multi_radius_range_list,
            )
        )
        total_multiplex_led_mask = th.cat(
            (
                self.DPC_multiplex_led_array_mask,
                self.dark_multiplex_led_array_mask,
            ),
            dim=0,
        )

        # determine the reconstruction size by multiplexed mask
        reconstruction_size = self.dataset.get_reconstruction_size(
            multiplex_led_array_mask=total_multiplex_led_mask
        )
        reconstruction_shape = (reconstruction_size, reconstruction_size)
        print(f"Dark recontruction shape: {reconstruction_shape}")

        # read pupil, shifts in F-space
        probe = self.dataset.get_pupil_mask()
        total_shifts_pair = self.dataset.total_shifts_pair

        # initial guess
        initial_est = th.ones(shape=reconstruction_shape, dtype=th.complex128)
        initial_est = th.fft.fft2(initial_est, norm="ortho")

        # upsampling initial guess if needed
        crop_op = pl.LinOpCrop2(
            in_shape=reconstruction_shape, crop_shape=initial_est.shape
        )
        initial_est = th.fft.ifftshift(
            crop_op.applyT(th.fft.fftshift(initial_est))
        )

        # obatin forward model and generate the measurements
        pr_model = phaseretrieval.MultiplexedPhaseRetrieval(
            probe=probe,
            multiplex_led_mask=total_multiplex_led_mask,
            shifts_pair=total_shifts_pair,
            reconstruct_shape=reconstruction_shape,
        )
        total_y = pr_model.forward(
            th.fft.fft2(np.array(simu_img), norm="ortho")
        )

        # algorithm solver
        ppr_method = algos.PerturbativePhase(pr_model)
        if dark_lr is not None:
            x_est = ppr_method.iterate_GradientDescent(
                y=total_y,
                initial_est=initial_est,
                n_iter=dark_n_iter,
                linear_n_iter=dark_linear_n_iter,
                lr=dark_lr,
            )
        else:
            x_est = ppr_method.iterate_ConjugateGradientDescent(
                y=total_y,
                x0=initial_est,
                n_iter=dark_n_iter,
                linear_n_iter=dark_linear_n_iter,
            )

        x_est = th.fft.ifft2(x_est, norm="ortho")

        return x_est

    def FPM(
        self,
        groundtruth_size,
        radius: float,
        n_iter=5,
        lr=1,
        centre=[0, 0],
        NA_range: Optional[list] = None,
        for_loop_or_not: bool = False,
    ):
        print("FPM Start \n----------------------")
        # load groundtruth
        simu_img = self.select_simulation_range(
            groundtruth_size=groundtruth_size, centre=centre
        )

        # shift in F-space and pupil
        img_idx_array, shifts_pair = self.dataset.select_FPM_image_by_NA(
            radius=radius
        )
        probe = np.array(self.dataset.get_pupil_mask())

        # define reconstruction size from LED index
        reconstruction_res = self.dataset.get_reconstruction_size(
            select_led_index_array=img_idx_array
        )
        reconstruction_shape = (reconstruction_res, reconstruction_res)
        print(f"recontruction size: {reconstruction_shape}")

        # forward model and generate measurements
        pr_model = phaseretrieval.FourierPtychography2d(
            probe=probe,
            shifts_pair=shifts_pair,
            reconstruct_shape=reconstruction_shape,
        )
        y = pr_model.forward(th.fft.fft2(simu_img, norm="ortho"))

        # initial guess
        initial_est = th.ones(shape=reconstruction_shape, dtype=th.complex128)
        initial_est = th.fft.fft2(initial_est, norm="ortho")

        # algorithm solver
        loss_function = loss.loss_amplitude_based(epsilon=1e-4)
        gd_method = algos.GradientDescent(
            pr_model, loss_func=loss_function, line_search=None
        )
        x_est = gd_method.iterate(
            y=y, initial_est=initial_est, n_iter=n_iter, lr=lr
        )
        x_est = th.fft.ifft2(x_est, norm="ortho")

        return x_est


def select_simulation_range(
    size: int,
    center: tuple[int, int] = (0, 0),
):
    img = skimage.data.camera()
    img = (img - np.min(img)) / (np.max(img) - np.min(img)) - 0.5
    img = np.exp((1j * img))

    v_center = img.shape[0] // 2 - center[0]
    h_center = img.shape[1] // 2 + center[1]

    groundtruth_img = img[
        v_center - size // 2 : v_center + size // 2,
        h_center - size // 2 : h_center + size // 2,
    ]

    return th.from_numpy(groundtruth_img).to(th.complex64).to(device)


if __name__ == "__main__":
    center = (256, 256)
    size = 256
    image = select_simulation_range(size=size, center=center)
    _simulation = DPC_and_darkField_solver(camera_size=100)
    centre = [50, 60]
    groundtruth_size = 256

    bright_angle_range = np.array([[0, 180], [90, 270]])
    pattern = "2 halves 90 degree"
    _simulation.Real_DPC(
        image,
        bright_field_angle_range=bright_angle_range,
        groundtruth_size=groundtruth_size,
        centre=centre,
        alpha=0.1,
    )

    bright_angle_range = np.array([[0, 180], [90, 270], [0, 361]])
    x_est = _simulation.BF_PPR(
        image,
        bright_field_angle_range=bright_angle_range,
        groundtruth_size=groundtruth_size,
        n_iter=100,
        linear_n_iter=120,
        lr=None,
        _lambda=0,
        alpha=0,
    )

    bright_angle_range = np.array([[0, 180], [90, 270], [0, 361]])
    multi_angle_range_list = [[0, 361], [0, 361]]

    inner_NA_factor = 0
    outer_NA_facter = 1.5
    inner_NA = _simulation.dataset.NA * inner_NA_factor
    outer_NA = _simulation.dataset.NA * outer_NA_facter
    inner_dark_radius = (
        (inner_NA * _simulation.dataset.led_d_z)
        / np.sqrt(1 - inner_NA**2)
        / _simulation.dataset.led_pitch
    )
    outer_dark_radius = (
        (outer_NA * _simulation.dataset.led_d_z)
        / np.sqrt(1 - outer_NA**2)
        / _simulation.dataset.led_pitch
    )
    single_radius_range = [
        np.sqrt(2 * (inner_dark_radius**2)),
        np.sqrt(2 * (outer_dark_radius**2)),
    ]

    inner_NA_factor = 1.5
    outer_NA_facter = 2
    inner_NA = _simulation.dataset.NA * inner_NA_factor
    outer_NA = _simulation.dataset.NA * outer_NA_facter
    inner_dark_radius = (
        (inner_NA * _simulation.dataset.led_d_z)
        / np.sqrt(1 - inner_NA**2)
        / _simulation.dataset.led_pitch
    )
    outer_dark_radius = (
        (outer_NA * _simulation.dataset.led_d_z)
        / np.sqrt(1 - outer_NA**2)
        / _simulation.dataset.led_pitch
    )
    second_radius_range = [
        np.sqrt(2 * (inner_dark_radius**2)),
        np.sqrt(2 * (outer_dark_radius**2)),
    ]

    multi_radius_range_list = [single_radius_range, second_radius_range]

    x_est = _simulation.DF_PPR(
        bright_field_angle_range=bright_angle_range,
        dark_multi_angle_range_list=multi_angle_range_list,
        dark_multi_radius_range_list=multi_radius_range_list,
        bright_n_iter=100,
        bright_linear_n_iter=100,
        bright_lr=None,
        dark_n_iter=100,
        dark_linear_n_iter=100,
        dark_lr=None,
        NA_range=[1, outer_NA_facter],
    )

    outer_NA_facter = 2.5
    outer_NA = _simulation.dataset.NA * outer_NA_facter
    outer_dark_radius = (
        (outer_NA * _simulation.dataset.led_d_z)
        / np.sqrt(1 - outer_NA**2)
        / _simulation.dataset.led_pitch
    )
    x_est = _simulation.FPM(
        radius=outer_dark_radius,
        n_iter=20,
        lr=1e-2,
        NA_range=[0, 2],
    )
