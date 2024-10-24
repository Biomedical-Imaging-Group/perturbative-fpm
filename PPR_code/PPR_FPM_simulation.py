import numpy as np
import time
import matplotlib.pyplot as plt
import skimage
import torch as th
import pyphaseretrieve.linop as pl
from pyphaseretrieve import algos, loss, phaseretrieval

device = th.device("mps")


class Microscope:
    def __init__(
        self,
        camera_size: int,
        wave_lambda: float = 0.514,
        na: float = 0.19,
        magnification: float = 8.1485,
        camera_pixel_Gsize: float = 5.5,
        led_pitch: float = 4_000,
        led_d_z: float = 67_500,
        led_dia_number: int = 25,
    ):
        # Experiment Setup Parameters Setting (distance and length unit: um)
        # Optical system
        self.wave_lambda = wave_lambda
        self.na = na
        # Camera system
        self.camera_size = camera_size
        self.magnification = magnification
        self.camera_pixel_Gsize = camera_pixel_Gsize
        # LED system
        self.led_pitch = led_pitch
        self.led_d_z = led_d_z
        self.led_dia_number = led_dia_number

        img_pixel_size = self.camera_pixel_Gsize / self.magnification
        fov = img_pixel_size * self.camera_size
        self.fourier_res = 1 / fov

        if 2 * self.na / self.wave_lambda > (1 / img_pixel_size) / 2:
            print("There is aliasing")

        camera_size_idx = th.arange(
            -np.floor(self.camera_size / 2),
            np.ceil(self.camera_size / 2),
            device=device,
        )
        temp_mask_h, temp_mask_v = th.meshgrid(
            camera_size_idx, camera_size_idx, indexing="ij"
        )
        pupil_radius = self.na / self.wave_lambda / self.fourier_res
        self.pupil_mask = pl.Fftshift() @ (
            th.sqrt(temp_mask_h**2 + temp_mask_v**2) <= pupil_radius)

        led_ra_size = np.floor(self.led_dia_number / 2)
        indices = th.arange(-led_ra_size, led_ra_size + 1, device=device)
        led_indices_h, led_indices_v = th.meshgrid(
            indices, indices, indexing="ij"
        )
        led_indices = th.stack((led_indices_v, led_indices_h), dim=-1)

        self.led_radii = th.sqrt((led_indices**2).sum(-1))
        self.led_angles = -th.flipud(
            th.fliplr(th.arctan2(led_indices_h, led_indices_v) - np.pi)
        )

        led_distances = led_indices * self.led_pitch
        sample_distances = th.sqrt(
            led_distances[..., 0] ** 2
            + led_distances[..., 1] ** 2
            + self.led_d_z**2
        )

        sin_theta = led_distances / sample_distances[:, :, None]

        self.shifts = th.round(sin_theta / self.wave_lambda / self.fourier_res)
        self.bf_mask = th.sqrt((sin_theta**2).sum(-1)) <= self.na

    # TODO these should probably not be inside of the microscope class
    def led_mask_by_angles(self, angle_ranges: th.Tensor) -> th.Tensor:
        masks = th.empty(
            (len(angle_ranges), *self.led_angles.shape),
            device=device,
            dtype=th.bool,
        )
        for i, angle_range in enumerate(angle_ranges):
            mask = (angle_range[0] <= self.led_angles) * (
                self.led_angles < angle_range[1]
            )
            masks[i] = mask

        return masks

    def bf_only(self, masks: th.Tensor) -> th.Tensor:
        return masks * self.bf_mask

    def led_mask_by_radii(self, radii_ranges: th.Tensor) -> th.Tensor:
        masks = th.empty(
            (len(radii_ranges), *self.led_angles.shape),
            device=device,
            dtype=th.bool,
        )
        for i, radius_range in enumerate(radii_ranges):
            mask = (radius_range[0] <= self.led_radii) * (
                self.led_radii < radius_range[1]
            )
            masks[i] = mask

        return masks

    def reconstruction_size(self, mask: th.Tensor) -> int:
        r_map = self.led_radii[None].expand(mask.shape)
        max_r = int(th.max(r_map[mask])) * self.led_pitch
        na_illu = max_r / np.sqrt(self.led_d_z**2 + max_r**2)
        synthetic_na = na_illu + self.na
        max_reconstruction_size = int(
            np.ceil(2 * synthetic_na / self.wave_lambda / self.fourier_res)
        )
        reconstruction_size = np.maximum(
            int(self.camera_size), int(max_reconstruction_size)
        )
        return reconstruction_size.item()


def DPC(image, masks: list[th.Tensor], alpha=0):
    # TODO figure out physics and what this is, adapt call signature to what
    # makes sense
    reconstruction_res = microscope.reconstruction_size(masks)
    all_shifts = [microscope.shifts[mask] for mask in masks]
    reconstruction_shape = (reconstruction_res, reconstruction_res)
    print(f"Real DPC recontruction shape: {reconstruction_shape}")
    pr_model = phaseretrieval.MultiplexedFourierPtychography(
        probe=probe,
        all_shifts=all_shifts,
        reconstruction_shape=reconstruction_shape,
    )
    y = pr_model.forward(pl.Fft2() @ image)

    crop = pl.Crop2(
        in_shape=(reconstruction_shape),
        crop_shape=(camera_size, camera_size),
    )

    numerator = th.zeros(probe.shape, dtype=th.complex64, device=device)
    denom = th.zeros(probe.shape, dtype=th.float32, device=device)
    probe_ = probe.to(th.int32)

    for i_m, shifts in enumerate(all_shifts):
        hm = (pl.Roll2(-shifts) @ probe_ - pl.Roll2(shifts) @ probe_).sum(
            1, keepdim=True
        )
        numerator += ((hm * 1j).conj() * pl.Ifftshift() @ crop
                      @ pl.Fftshift() @ pl.Fft2() @ y[:, i_m:i_m + 1])
        denom += th.abs(hm) ** 2

    return pl.Ifft2() @ (numerator / (denom + alpha))


# TODO alpha and lamda are unused atm
def PPR(image, masks, n_iter=1, linear_n_iter=1, alpha=0, lamda=0):
    res = microscope.reconstruction_size(masks)
    all_shifts = [microscope.shifts[mask] for mask in masks]
    rec_shape = (res, res)
    print(f"DPC recontruction shape: {rec_shape}")

    model = phaseretrieval.MultiplexedFourierPtychography(
        probe=probe,
        all_shifts=all_shifts,
        reconstruction_shape=rec_shape,
    )
    y = model.forward(pl.Fft2() @ image)

    # TODO we need to discuss how to properly separate all of this stuff
    # probably have a `residual` function in the model that takes the data as
    # argument. i dont think it makes sense to give the data into the
    # microscope class
    def f(x):
        return y - model.forward(x)

    f.jacobian = model.jacobian

    # TODO we could warm start CG with the previous solution if needed
    def solve(A, b):
        return th.view_as_complex(
            algos.conjugate_gradient(
                A,
                b,
                th.zeros_like(A.T @ b),
                n_iter=linear_n_iter,
                dim=(1, 2, 3, 4),
            )
        )

    # TODO remove hardcoded 1, 1
    x0 = pl.Fft2() @ th.ones(
        (1, 1, *rec_shape), dtype=th.complex64, device=device)
    start = time.time()
    x_est = algos.gauss_newton(f=f, x0=x0, n_iter=n_iter, solve=solve)
    print(f"PPR time: {time.time() - start}")
    return pl.Ifft2() @ x_est


def FPM(image, radius: float, n_iter=5, lr=1):
    select_led_index_map = microscope.led_radii <= radius
    shifts_pair = microscope.shifts[select_led_index_map][:, None]
    res = microscope.reconstruction_size(select_led_index_map[None])
    reconstruction_shape = (res, res)

    pr_model = phaseretrieval.FourierPtychography2d(
        probe=probe,
        shifts_pair=shifts_pair,
        reconstruction_shape=reconstruction_shape,
    )
    y = pr_model.forward(pl.Fft2() @ image)

    initial_est = th.ones(
        (1, 1, *reconstruction_shape), dtype=th.complex64, device=device
    )
    initial_est = pl.Fft2() @ initial_est

    loss_function = loss.loss_amplitude_based(epsilon=1e-4)
    gd_method = algos.GradientDescent(
        pr_model, loss_func=loss_function, line_search=None
    )
    x_est = gd_method.iterate(
        y=y, initial_est=initial_est, n_iter=n_iter, lr=lr
    )
    return pl.Ifft2() @ x_est


if __name__ == "__main__":
    size = 256
    center = (50, 60)
    img = skimage.data.camera()
    img = (img - np.min(img)) / (np.max(img) - np.min(img)) - 0.5
    img = np.exp(1j * img)

    v_center = img.shape[0] // 2 - 60
    h_center = img.shape[1] // 2 + 50
    img = img[
        v_center - size // 2: v_center + size // 2,
        h_center - size // 2: h_center + size // 2,
    ]
    image = th.from_numpy(img).to(th.complex64).to(device)[None, None]

    camera_size = 100
    microscope = Microscope(camera_size=camera_size)
    probe = microscope.pupil_mask[None, None]
    groundtruth_size = 256

    # TODO these produce the wrong thing for now because of how the angles are
    # constructed
    angle_ranges = np.array([[0.0, np.pi], [np.pi / 2, 3 * np.pi / 2]])
    dpc_masks = microscope.led_mask_by_angles(angle_ranges=angle_ranges)
    dpc_masks = microscope.bf_only(dpc_masks)
    x_est = DPC(image, dpc_masks, alpha=0.1)
    plt.figure()
    plt.imshow(x_est.real.cpu().numpy()[0, 0])

    angle_ranges = np.array(
        [[0, np.pi], [np.pi / 2, 3 * np.pi / 2], [0, 2 * np.pi]]
    )
    ppr_bf_masks = microscope.led_mask_by_angles(angle_ranges=angle_ranges)
    ppr_bf_masks = microscope.bf_only(ppr_bf_masks)
    x_est = PPR(image, ppr_bf_masks, n_iter=20, linear_n_iter=20)
    plt.figure()
    plt.imshow(th.angle(x_est).cpu().numpy()[0, 0])

    def radius(factor):
        na = microscope.na * factor
        r = na * microscope.led_d_z / np.sqrt(1 - na**2) / microscope.led_pitch
        return np.sqrt(2 * r**2)

    radii_ranges = [
        [radius(inner_factor), radius(outer_factor)]
        for (inner_factor, outer_factor) in [(0, 1.5), (1.5, 2)]
    ]

    ppr_df_masks = microscope.led_mask_by_radii(radii_ranges=radii_ranges)
    ppr_df_masks[microscope.bf_mask[None].expand(2, 25, 25)] = False
    ppr_masks = th.cat((ppr_bf_masks, ppr_df_masks), dim=0)
    x_est = PPR(image, ppr_masks, n_iter=10, linear_n_iter=10)
    plt.figure()
    plt.imshow(th.angle(x_est).cpu().numpy()[0, 0])

    print("FPM Start \n----------------------")
    outer_factor = 2.5
    outer_dark_radius = radius(outer_factor)

    x_est = FPM(image, radius=outer_dark_radius, n_iter=20, lr=1e-2)
    plt.figure()
    plt.imshow(th.angle(x_est).cpu().numpy()[0, 0])
    plt.show()
