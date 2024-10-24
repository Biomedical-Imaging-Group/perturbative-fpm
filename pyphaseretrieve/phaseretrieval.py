import torch as th
import numpy as np
import pyphaseretrieve.linop as pl


class Ptychography1d:
    def __init__(self, probe: th.Tensor, shifts: th.Tensor | None = None, n_img: int = 10):
        self.probe = probe
        self.probe_shape = probe.shape

        if shifts is not None:
            self.n_img = len(shifts)
            self.shifts = shifts
        else:
            self.n_img = n_img
            self.shifts = self.get_auto_shifts()

        self.linop = self.get_forward_model()
        self.in_shape = self.linop.in_shape

    def get_auto_shifts(self) -> th.Tensor:
        probe_dia = np.count_nonzero(self.probe)
        start_shift = -(self.probe_shape[0] - probe_dia) // 2
        end_shift = (self.probe_shape[0] - probe_dia) // 2
        shifts = np.linspace(start_shift, end_shift, self.n_img)
        return shifts

    def get_forward_model(self) -> pl.LinOp:
        op_fft = pl.Fft()
        op_probe = pl.Mul(self.probe)
        linop = pl.Stack(
            [
                op_fft @ op_probe @ pl.Roll(self.shifts[i_probe])
                for i_probe in range(self.n_img)
            ]
        )
        return linop

    def get_probe_overlap_array(self) -> th.Tensor:
        overlap_img = np.zeros(shape=self.probe_shape)
        for i_probe in range(self.n_img):
            roll_linop = pl.Roll(-self.shifts[i_probe])
            overlap_img = overlap_img + roll_linop.apply(self.probe)
        return overlap_img

    def get_overlap_rate(self) -> float:
        probe_dia = np.count_nonzero(self.probe)
        step_size = np.abs(self.shifts[0] - self.shifts[1])
        overlap = 1 - step_size / probe_dia
        return overlap


class FourierPtychography2d:
    def __init__(
        self,
        probe,
        shifts_pair: th.Tensor,
        reconstruction_shape: tuple[int, int] | None = None,
        n_img: int = 25,
    ):
        self.probe = probe
        self.reconstruction_shape = reconstruction_shape or self.probe.shape
        self.n_img = shifts_pair.shape[0]
        self.shifts = shifts_pair

        self.linop = self.get_forward_model()
        self.in_shape = self.linop.in_shape

    def get_auto_shifts_pair(self) -> th.Tensor:
        shift_probe = np.fft.fftshift(self.probe)
        probe_center_row = shift_probe[int(self.probe.shape[0] // 2)]
        probe_dia = np.count_nonzero(probe_center_row)

        start_shift = -(self.reconstruction_shape[0] - probe_dia) // 2
        end_shift = (self.reconstruction_shape[0] - probe_dia) // 2
        side_n_img = int(np.sqrt(self.n_img))
        shifts = np.linspace(start_shift, end_shift, side_n_img).astype(int)
        shifts_h, shifts_v = np.meshgrid(shifts, shifts)
        shifts_pair = np.concatenate(
            [shifts_v.reshape(self.n_img, 1), shifts_h.reshape(self.n_img, 1)],
            axis=1,
        )
        return shifts_pair

    # TODO the parallelism is extremely suboptimal here as it is essentially
    # the same code as the multiplexed one which parallelizes over the LEDs
    # Here, each forward only has one LED, so we are in the worst case..
    def get_forward_model(self) -> pl.LinOp:
        return pl.Stack(
            [
                pl.Ifft2()
                @ pl.Mul(self.probe)
                @ pl.Ifftshift()
                @ pl.Crop2(
                    in_shape=self.reconstruction_shape,
                    crop_shape=self.probe.shape[2:],
                )
                @ pl.Fftshift()
                @ pl.Roll2(self.shifts[i_probe])
                for i_probe in range(self.n_img)
            ]
        )

    def forward(self, x: th.Tensor) -> th.Tensor:
        return (
            th.abs((self.linop @ x) * (self.probe.shape[0] / x.shape[0])) ** 2
        )

    def apply(self, x):
        return self.linop.apply(x)

    def applyT(self, x):
        return self.linop.applyT(x)

    def get_overlap_rate(self) -> float:
        """self-defined shifts_pair might cause error"""
        shift_probe = np.fft.fftshift(self.probe)
        probe_center_row = shift_probe[int(self.probe.shape[0] // 2)]
        probe_dia = np.count_nonzero(probe_center_row)
        probe_radius = probe_dia // 2
        step_size = th.abs(self.shifts_pair[0][1] - self.shifts_pair[1][1])
        if step_size > (probe_radius * 2):
            return 0
        else:
            circ_sector = (
                2
                * (np.arccos(step_size / 2 / probe_radius) / (2 * np.pi))
                * np.pi
                * probe_radius**2
            )
            tria_area = (
                step_size / 2 * np.sqrt(probe_radius**2 - (step_size / 2) ** 2)
            )
            overlap_rate = (
                2 * (circ_sector - tria_area) / np.pi / (probe_radius**2)
            )
            return overlap_rate

    def get_probe_overlap_map(self) -> th.Tensor:
        pad_size = self.reconstruction_shape[0] - self.probe.shape[0]
        shift_probe = np.fft.fftshift(self.probe)
        shift_probe = np.pad(
            shift_probe,
            (int(np.floor(pad_size / 2)), int(np.ceil(pad_size / 2))),
            mode="constant",
        )

        overlap_img = np.zeros_like(shift_probe)
        for i_probe in range(self.n_img):
            roll_linop = pl.Roll2(
                -self.shifts_pair[i_probe, 0], -self.shifts_pair[i_probe, 1]
            )
            overlap_img = overlap_img + roll_linop.apply(shift_probe)
        return overlap_img


class XRay_Ptychography2d:
    def __init__(
        self,
        probe,
        shifts_pair: th.Tensor | None = None,
        reconstruction_shape: tuple[int, int] | None = None,
        n_img: int = 25,
    ):
        """shifts_pair is defined as [v_shifts,h_shifts]"""
        self.probe = probe
        self.probe_shape = probe.shape

        if reconstruction_shape is not None:
            self.reconstruction_shape = reconstruction_shape
        else:
            self.reconstruction_shape = self.probe_shape

        if shifts_pair is not None:
            assert (
                shifts_pair.ndim == 2
            ), "shifts_map dimension should be (n,2)"
            self.n_img = shifts_pair.shape[0]
            self.shifts_pair = shifts_pair
        else:
            assert (
                int(np.sqrt(n_img)) ** 2 == n_img
            ), "n_img need to be perfect square"
            self.n_img = n_img
            self.shifts_pair = self.get_auto_shifts_pair()

        self.linop = self.get_forward_model()
        self.in_shape = self.linop.in_shape

    def get_auto_shifts_pair(self) -> th.Tensor:
        shift_probe = np.fft.fftshift(self.probe)
        probe_center_row = shift_probe[int(self.probe_shape[0] // 2)]
        probe_dia = np.count_nonzero(probe_center_row)

        start_shift = -(self.reconstruction_shape[0] - probe_dia) // 2
        end_shift = (self.reconstruction_shape[0] - probe_dia) // 2
        side_n_img = int(np.sqrt(self.n_img))
        shifts = np.linspace(start_shift, end_shift, side_n_img).astype(int)
        shifts_h, shifts_v = np.meshgrid(shifts, shifts)
        shifts_pair = np.concatenate(
            [shifts_v.reshape(self.n_img, 1), shifts_h.reshape(self.n_img, 1)],
            axis=1,
        )
        return shifts_pair

    def get_forward_model(self) -> pl.LinOp:
        op_fft2 = pl.Fft2()
        op_ifftshift = pl.Ifftshift()
        op_fcrop = pl.Crop2(
            in_shape=self.reconstruction_shape, crop_shape=self.probe_shape
        )
        op_probe = pl.Mul(self.probe)
        linop = pl.Stack(
            [
                op_ifftshift
                @ op_fft2
                @ op_probe
                @ op_fcrop
                @ pl.Roll2_PadZero(
                    self.shifts_pair[i_probe, 0], self.shifts_pair[i_probe, 1]
                )
                for i_probe in range(self.n_img)
            ]
        )
        return linop

    def get_linop_list(self):
        return self.linop.List

    def get_probe_overlap_map(self) -> th.Tensor:
        op_fcrop = pl.Crop2(
            in_shape=self.reconstruction_shape, crop_shape=self.probe_shape
        )
        shift_probe = op_fcrop.applyT(self.probe)

        overlap_img = np.zeros_like(shift_probe)
        for i_probe in range(self.n_img):
            roll_linop = pl.Roll2_PadZero(
                -self.shifts_pair[i_probe, 0], -self.shifts_pair[i_probe, 1]
            )
            overlap_img = overlap_img + roll_linop.apply(shift_probe)
        return overlap_img


class MultiplexedFourierPtychography:
    def __init__(
        self,
        probe,
        all_shifts: list[th.Tensor],
        reconstruction_shape: tuple[int, int],
    ):
        self.probe = probe
        self.all_shifts = all_shifts
        self.n_leds = [shifts.shape[0] for shifts in all_shifts]
        self.reconstruction_shape = reconstruction_shape
        self.forwards = [
            pl.Ifft2()
            @ pl.Mul(self.probe)
            @ pl.Ifftshift()
            @ pl.Crop2(
                in_shape=reconstruction_shape, crop_shape=self.probe.shape[2:]
            )
            @ pl.Fftshift()
            @ pl.Roll2(shifts)
            for shifts in all_shifts
        ]

    def forward(self, x: th.Tensor):
        y = th.empty(
            (x.shape[0], len(self.all_shifts), *self.probe.shape[2:]),
            device=x.device,
        )
        for i, forward in enumerate(self.forwards):
            y[:, i] = (th.abs(forward @ x) ** 2).sum(1)

        return y

    def jacobian(self, x: th.Tensor):
        return pl.Stack([
            pl.SumReduce(1, n_leds)
            @ pl.RealPartExpand(2 * pl.Mul((forward @ x).conj()) @ forward)
            # TODO better name; this is not the forward in class scope!
            for forward, n_leds in zip(self.forwards, self.n_leds)
        ])

    def get_probe_overlap_map(self) -> th.Tensor:
        pad_size = self.reconstruction_shape[0] - self.probe.shape[0]
        shift_probe = np.fft.fftshift(self.probe)
        shift_probe = np.pad(
            shift_probe,
            (int(np.floor(pad_size / 2)), int(np.ceil(pad_size / 2))),
            mode="constant",
        )

        overlap_img = np.zeros_like(shift_probe)
        for _, i_mask in enumerate(self.multiplex_led_mask):
            for idx, mask_item in enumerate(i_mask):
                if mask_item >= 1:
                    roll_linop = pl.Roll2(
                        -self.shifts_pair[idx, 0], -self.shifts_pair[idx, 1]
                    )
                    overlap_img = overlap_img + roll_linop @ shift_probe
        return overlap_img
