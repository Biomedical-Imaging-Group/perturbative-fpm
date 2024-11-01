import torch as th
import matplotlib.pyplot as plt
import numpy as np
import pyphaseretrieve.linop as pl


class Microscope:
    def __init__(
        self,
        led_positions: th.Tensor,
        camera_size: int,
        lamda: float = 0.514,
        na: float = 0.19,
        magnification: float = 8.1485,
        pixel_size: float = 5.5,
    ):
        # Experiment Setup Parameters Setting (distance and length unit: um)
        # Optical system
        self.lamda = lamda
        self.na = na
        # Camera system
        self.camera_size = camera_size
        self.magnification = magnification

        img_pixel_size = pixel_size / self.magnification
        fov = img_pixel_size * self.camera_size
        self.fourier_res = 1 / fov

        if 2 * self.na / self.lamda > (1 / img_pixel_size) / 2:
            print("There is aliasing")

        camera_size_idx = th.arange(
            -np.floor(self.camera_size / 2),
            np.ceil(self.camera_size / 2),
            device=led_positions.device,
        )
        temp_mask_h, temp_mask_v = th.meshgrid(
            camera_size_idx, camera_size_idx, indexing="ij"
        )
        pupil_radius = self.na / self.lamda / self.fourier_res
        self.pupil_mask = pl.Fftshift() @ (
            th.sqrt(temp_mask_h**2 + temp_mask_v**2) <= pupil_radius)

        self.led_positions = led_positions
        sin_theta = led_positions[:, 0:2] / \
            th.sqrt((led_positions ** 2).sum(-1, keepdims=True))
        self.led_na = (sin_theta ** 2).sum(-1).sqrt()
        self.shifts = sin_theta / self.lamda / self.fourier_res

    # TODO check correctness, looks very weird; taken from:
    # self.bf_mask = th.sqrt((sin_theta**2).sum(-1)) <= self.na
    def is_brightfield_led(self, position: th.Tensor) -> th.Tensor:
        tmp = position[:2] / (position ** 2).sum().sqrt()
        angle = th.angle(tmp[0] + 1j * tmp[1])
        return angle <= self.na

    # TODO check correctness
    # TODO change name, something like `max_unaliased_grid`?
    def reconstruction_size(self, indices: th.Tensor) -> int:
        na_illu = self.led_na[indices].max()
        na = na_illu + self.na
        return np.maximum(
            int(self.camera_size), int(
                th.ceil(2 * na / self.lamda / self.fourier_res)
            )
        )


class MultiplexedFourierPtychography:
    def __init__(
        self,
        microsope: Microscope,
        indices: list[th.Tensor],
        shape: tuple[int, int],
        exact_phase: bool = True,
    ):
        self.probe = microsope.pupil_mask[None, None]
        self.n_leds = [len(index) for index in indices]
        self.n_patterns = len(indices)

        phase_shift_ffts = [
            pl.Fft2() @ pl.PhaseShift(microsope.shifts[index], shape)
            for index in indices
        ] if exact_phase else [
            pl.Roll2(th.round(microsope.shifts[index]).to(th.int64)) @ pl.Fft2()
            for index in indices
        ]

        self.forwards = [
            pl.Ifft2()
            @ pl.Mul(self.probe)
            @ pl.Ifftshift()
            @ pl.Crop2(in_shape=shape, crop_shape=self.probe.shape[2:])
            @ pl.Fftshift()
            @ ps_fft
            for ps_fft in phase_shift_ffts
        ]

    def forward(self, x: th.Tensor):
        y = th.empty(
            (x.shape[0], self.n_patterns, *self.probe.shape[2:]),
            device=x.device,
        )
        for i, forward_ in enumerate(self.forwards):
            y[:, i] = (th.abs(forward_ @ x) ** 2).sum(1)

        return y

    def jacobian(self, x: th.Tensor):
        return pl.Stack([
            pl.SumReduce(1, n_leds)
            @ pl.RealPartExpand() @ pl.Mul(2 * (forward @ x).conj()) @ forward
            for forward, n_leds in zip(self.forwards, self.n_leds)
        ])
