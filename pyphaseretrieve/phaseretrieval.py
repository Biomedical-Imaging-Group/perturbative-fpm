from abc import abstractmethod
from typing import Optional
import numpy as np
import pyphaseretrieve.linop as pl


class PhaseRetrievalBase:
    def __init__(self, linop: pl.BaseLinOp):
        self.linop = linop
        self.in_shape = None
        self.probe_shape = None

    def apply_ModularSquare(self, x: np.ndarray):
        return (
            np.abs(self.linop.apply(x) * (self.probe_shape[0] / x.shape[0]))
            ** 2
        )

    def apply(self, x):
        return self.linop.apply(x)

    def applyT(self, x):
        return self.linop.applyT(x)

    @abstractmethod
    def get_forward_model(self):
        pass

    def get_perturbative_model(self, x, method: str):
        pass


class Ptychography1d(PhaseRetrievalBase):
    def __init__(self, probe, shifts: np.ndarray = None, n_img: int = 10):
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

    @abstractmethod
    def get_auto_shifts(self) -> np.ndarray:
        probe_dia = np.count_nonzero(self.probe)
        start_shift = -(self.probe_shape[0] - probe_dia) // 2
        end_shift = (self.probe_shape[0] - probe_dia) // 2
        shifts = np.linspace(start_shift, end_shift, self.n_img)
        return shifts

    def get_forward_model(self) -> pl.BaseLinOp:
        op_fft = pl.LinOpFFT()
        op_probe = pl.LinOpMul(self.probe)
        linop = pl.StackLinOp(
            [
                op_fft @ op_probe @ pl.LinOpRoll(self.shifts[i_probe])
                for i_probe in range(self.n_img)
            ]
        )
        return linop

    def get_perturbative_model(self, x_est, method: str):
        if method == "GradientDescent":
            return self.get_perturbative_GradientDescent_model(x_est=x_est)
        elif method == "ConjugateGradientDescent":
            return self.get_perturbative_ConjugateGradientDescent_model(
                x_est=x_est
            )

    def get_perturbative_GradientDescent_model(self, x_est):
        out_field = self.apply(x_est)
        perturbative_model = (
            2 * pl.LinOpReal() @ pl.LinOpMul(out_field.conj()) @ self.linop
        )
        return perturbative_model, None

    def get_perturbative_ConjugateGradientDescent_model(self, x_est):
        out_field = self.apply(x_est)
        perturbative_model = pl.LinOp_RealPartExpand(
            2 * pl.LinOpMul(out_field.conj()) @ self.linop
        )
        return perturbative_model, None

    def get_probe_overlap_array(self) -> np.ndarray:
        overlap_img = np.zeros(shape=self.probe_shape)
        for i_probe in range(self.n_img):
            roll_linop = pl.LinOpRoll(-self.shifts[i_probe])
            overlap_img = overlap_img + roll_linop.apply(self.probe)
        return overlap_img

    def get_overlap_rate(self) -> float:
        probe_dia = np.count_nonzero(self.probe)
        step_size = np.abs(self.shifts[0] - self.shifts[1])
        overlap = 1 - step_size / probe_dia
        return overlap


class FourierPtychography2d(PhaseRetrievalBase):
    def __init__(
        self,
        probe,
        shifts_pair: np.ndarray = None,
        reconstruct_shape: Optional[int] = None,
        n_img: int = 25,
    ):
        """shifts_pair is defined as [v_shifts,h_shifts]"""
        self.probe = probe
        self.probe_shape = probe.shape

        if reconstruct_shape is not None:
            self.reconstruct_shape = reconstruct_shape
        else:
            self.reconstruct_shape = self.probe_shape

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

    @abstractmethod
    def get_auto_shifts_pair(self) -> np.ndarray:
        shift_probe = np.fft.fftshift(self.probe)
        probe_center_row = shift_probe[int(self.probe_shape[0] // 2)]
        probe_dia = np.count_nonzero(probe_center_row)

        start_shift = -(self.reconstruct_shape[0] - probe_dia) // 2
        end_shift = (self.reconstruct_shape[0] - probe_dia) // 2
        side_n_img = int(np.sqrt(self.n_img))
        shifts = np.linspace(start_shift, end_shift, side_n_img).astype(int)
        shifts_h, shifts_v = np.meshgrid(shifts, shifts)
        shifts_pair = np.concatenate(
            [shifts_v.reshape(self.n_img, 1), shifts_h.reshape(self.n_img, 1)],
            axis=1,
        )
        return shifts_pair

    def get_forward_model(self) -> pl.BaseLinOp:
        op_ifft2 = pl.LinOpIFFT2()
        op_fftshift = pl.LinOpFFTSHIFT()
        op_ifftshift = pl.LinOpIFFTSHIFT()
        op_fcrop = pl.LinOpCrop2(
            in_shape=self.reconstruct_shape, crop_shape=self.probe_shape
        )
        op_probe = pl.LinOpMul(self.probe)
        linop = pl.StackLinOp(
            [
                op_ifft2
                @ op_probe
                @ op_ifftshift
                @ op_fcrop
                @ op_fftshift
                @ pl.LinOpRoll2(
                    self.shifts_pair[i_probe, 0], self.shifts_pair[i_probe, 1]
                )
                for i_probe in range(self.n_img)
            ]
        )
        return linop

    def get_perturbative_model(self, x_est, method: str):
        if method == "GradientDescent":
            return self.get_perturbative_GradientDescent_model(x_est=x_est)
        elif method == "ConjugateGradientDescent":
            return self.get_perturbative_ConjugateGradientDescent_model(
                x_est=x_est
            )

    def get_perturbative_GradientDescent_model(self, x_est):
        out_field = self.apply(x_est)
        perturbative_model = (
            2 * pl.LinOpReal() @ pl.LinOpMul(out_field.conj()) @ self.linop
        )
        return perturbative_model, None

    def get_perturbative_ConjugateGradientDescent_model(self, x_est):
        out_field = self.apply(x_est)
        perturbative_model = pl.LinOp_RealPartExpand(
            2 * pl.LinOpMul(out_field.conj()) @ self.linop
        )
        return perturbative_model, None

    def get_overlap_rate(self) -> float:
        """self-defined shifts_pair might cause error"""
        shift_probe = np.fft.fftshift(self.probe)
        probe_center_row = shift_probe[int(self.probe_shape[0] // 2)]
        probe_dia = np.count_nonzero(probe_center_row)
        probe_radius = probe_dia // 2
        step_size = np.abs(self.shifts_pair[0][1] - self.shifts_pair[1][1])
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

    def get_probe_overlap_map(self) -> np.ndarray:
        pad_size = self.reconstruct_shape[0] - self.probe_shape[0]
        shift_probe = np.fft.fftshift(self.probe)
        shift_probe = np.pad(
            shift_probe,
            (int(np.floor(pad_size / 2)), int(np.ceil(pad_size / 2))),
            mode="constant",
        )

        overlap_img = np.zeros_like(shift_probe)
        for i_probe in range(self.n_img):
            roll_linop = pl.LinOpRoll2(
                -self.shifts_pair[i_probe, 0], -self.shifts_pair[i_probe, 1]
            )
            overlap_img = overlap_img + roll_linop.apply(shift_probe)
        return overlap_img


class XRay_Ptychography2d(PhaseRetrievalBase):
    def __init__(
        self,
        probe,
        shifts_pair: np.ndarray = None,
        reconstruct_shape: Optional[tuple] = None,
        n_img: int = 25,
    ):
        """shifts_pair is defined as [v_shifts,h_shifts]"""
        self.probe = probe
        self.probe_shape = probe.shape

        if reconstruct_shape is not None:
            self.reconstruct_shape = reconstruct_shape
        else:
            self.reconstruct_shape = self.probe_shape

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

    @abstractmethod
    def get_auto_shifts_pair(self) -> np.ndarray:
        shift_probe = np.fft.fftshift(self.probe)
        probe_center_row = shift_probe[int(self.probe_shape[0] // 2)]
        probe_dia = np.count_nonzero(probe_center_row)

        start_shift = -(self.reconstruct_shape[0] - probe_dia) // 2
        end_shift = (self.reconstruct_shape[0] - probe_dia) // 2
        side_n_img = int(np.sqrt(self.n_img))
        shifts = np.linspace(start_shift, end_shift, side_n_img).astype(int)
        shifts_h, shifts_v = np.meshgrid(shifts, shifts)
        shifts_pair = np.concatenate(
            [shifts_v.reshape(self.n_img, 1), shifts_h.reshape(self.n_img, 1)],
            axis=1,
        )
        return shifts_pair

    def get_forward_model(self) -> pl.BaseLinOp:
        op_fft2 = pl.LinOpFFT2()
        op_ifftshift = pl.LinOpIFFTSHIFT()
        op_fcrop = pl.LinOpCrop2(
            in_shape=self.reconstruct_shape, crop_shape=self.probe_shape
        )
        op_probe = pl.LinOpMul(self.probe)
        linop = pl.StackLinOp(
            [
                op_ifftshift
                @ op_fft2
                @ op_probe
                @ op_fcrop
                @ pl.LinOpRoll2_PadZero(
                    self.shifts_pair[i_probe, 0], self.shifts_pair[i_probe, 1]
                )
                for i_probe in range(self.n_img)
            ]
        )
        return linop

    def get_linop_list(self):
        return self.linop.LinOpList

    def get_perturbative_model(self, x_est, method: str):
        if method == "GradientDescent":
            return self.get_perturbative_GradientDescent_model(x_est=x_est)
        elif method == "ConjugateGradientDescent":
            return self.get_perturbative_ConjugateGradientDescent_model(
                x_est=x_est
            )

    def get_perturbative_GradientDescent_model(self, x_est):
        out_field = self.apply(x_est)
        perturbative_model = (
            2 * pl.LinOpReal() @ pl.LinOpMul(out_field.conj()) @ self.linop
        )
        return perturbative_model, None

    def get_perturbative_ConjugateGradientDescent_model(self, x_est):
        out_field = self.apply(x_est)
        perturbative_model = pl.LinOp_RealPartExpand(
            2 * pl.LinOpMul(out_field.conj()) @ self.linop
        )
        return perturbative_model, None

    def get_probe_overlap_map(self) -> np.ndarray:
        op_fcrop = pl.LinOpCrop2(
            in_shape=self.reconstruct_shape, crop_shape=self.probe_shape
        )
        shift_probe = op_fcrop.applyT(self.probe)

        overlap_img = np.zeros_like(shift_probe)
        for i_probe in range(self.n_img):
            roll_linop = pl.LinOpRoll2_PadZero(
                -self.shifts_pair[i_probe, 0], -self.shifts_pair[i_probe, 1]
            )
            overlap_img = overlap_img + roll_linop.apply(shift_probe)
        return overlap_img


class MultiplexedPhaseRetrieval(PhaseRetrievalBase):
    def __init__(
        self,
        probe,
        multiplex_led_mask: np.ndarray,
        shifts_pair: np.ndarray = None,
        reconstruct_shape=None,
    ):
        """shifts_pair is defined as [v_shifts,h_shifts]"""
        self.probe = probe
        self.probe_shape = probe.shape
        self.multiplex_led_mask = multiplex_led_mask

        if reconstruct_shape is not None:
            self.reconstruct_shape = reconstruct_shape
        else:
            self.reconstruct_shape = self.probe_shape

        assert shifts_pair.ndim == 2, "shifts_map dimension should be (n,2)"
        self.n_img = self.multiplex_led_mask.shape[0]
        self.shifts_pair = shifts_pair

        self.total_linop_list = self.get_total_linop_list()
        self.in_shape = reconstruct_shape

    def get_total_linop_list(self) -> list:
        op_ifft2 = pl.LinOpIFFT2()
        op_fftshift = pl.LinOpFFTSHIFT()
        op_ifftshift = pl.LinOpIFFTSHIFT()
        op_fcrop = pl.LinOpCrop2(
            in_shape=self.reconstruct_shape, crop_shape=self.probe_shape
        )
        op_probe = pl.LinOpMul(self.probe)

        total_linop_list = []
        for i_probe in range(self.shifts_pair.shape[0]):
            _linop = (
                op_ifft2
                @ op_probe
                @ op_ifftshift
                @ op_fcrop
                @ op_fftshift
                @ pl.LinOpRoll2(
                    self.shifts_pair[i_probe, 0], self.shifts_pair[i_probe, 1]
                )
            )
            total_linop_list.append(_linop)
        return total_linop_list

    def apply_ModularSquare(self, x):
        y_est = None
        for _, i_mask in enumerate(self.multiplex_led_mask):
            single_y = 0
            for idx, mask_item in enumerate(i_mask):
                if (mask_item != False) and (mask_item != 0):
                    single_y += (
                        np.abs(
                            self.total_linop_list[idx].apply(x)
                            * (self.probe_shape[0] / x.shape[0])
                        )
                        ** 2
                        * mask_item
                    )
            if y_est is None:
                y_est = np.copy(single_y)
            else:
                y_est = np.concatenate((y_est, single_y), axis=0)
        return y_est

    def apply(self, x):
        raise NameError("No apply method in MultiplexedPhaseRetrieval")

    def applyT(self, x):
        raise NameError("No applyT method in MultiplexedPhaseRetrieval")

    def get_perturbative_model(self, x_est, method: str):
        if method == "GradientDescent":
            return self.get_perturbative_GradientDescent_model(x_est=x_est)
        elif method == "ConjugateGradientDescent":
            return self.get_perturbative_ConjugateGradientDescent_model(
                x_est=x_est
            )

    def get_perturbative_GradientDescent_model(self, x_est):
        perturbative_model_list = []
        for _, i_mask in enumerate(self.multiplex_led_mask):
            _perturbative_model = None
            for idx, mask_item in enumerate(i_mask):
                if (mask_item != False) and (mask_item != 0):
                    _out_field = self.total_linop_list[idx].apply(x_est)
                    if _perturbative_model is None:
                        _perturbative_model = (
                            2
                            * pl.LinOpReal()
                            @ pl.LinOpMul(_out_field.conj())
                            @ self.total_linop_list[idx]
                            * mask_item
                        )
                    else:
                        _perturbative_model += (
                            2
                            * pl.LinOpReal()
                            @ pl.LinOpMul(_out_field.conj())
                            @ self.total_linop_list[idx]
                            * mask_item
                        )
            perturbative_model_list.append(_perturbative_model)
        perturbative_model = pl.StackLinOp(perturbative_model_list)
        return perturbative_model, perturbative_model_list

    def get_perturbative_ConjugateGradientDescent_model(self, x_est):
        perturbative_model_list = []
        for _, i_mask in enumerate(self.multiplex_led_mask):
            _perturbative_model = None
            for idx, mask_item in enumerate(i_mask):
                if (mask_item != False) and (mask_item != 0):
                    _out_field = self.total_linop_list[idx].apply(x_est)
                    if _perturbative_model is None:
                        _perturbative_model = (
                            pl.LinOp_RealPartExpand(
                                2
                                * pl.LinOpMul(_out_field.conj())
                                @ self.total_linop_list[idx]
                            )
                            * mask_item
                        )
                    else:
                        _perturbative_model += (
                            pl.LinOp_RealPartExpand(
                                2
                                * pl.LinOpMul(_out_field.conj())
                                @ self.total_linop_list[idx]
                            )
                            * mask_item
                        )
            perturbative_model_list.append(_perturbative_model)
        perturbative_model = pl.StackLinOp(perturbative_model_list)
        return perturbative_model, perturbative_model_list

    def get_probe_overlap_map(self) -> np.ndarray:
        pad_size = self.reconstruct_shape[0] - self.probe_shape[0]
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
                    roll_linop = pl.LinOpRoll2(
                        -self.shifts_pair[idx, 0], -self.shifts_pair[idx, 1]
                    )
                    overlap_img = overlap_img + roll_linop.apply(shift_probe)
        return overlap_img
