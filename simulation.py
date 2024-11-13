import numpy as np
import os
from pyphaseretrieve.phaseretrieval import Microscope
from pathlib import Path
import utils
import skimage
import torch as th
import pyphaseretrieve.phaseretrieval as pp


def led_indices_by_angles(
    positions: th.Tensor, angle_ranges: th.Tensor
) -> list[th.Tensor]:
    sin_theta = positions[:, 0:2] / th.sqrt((positions**2).sum(-1, keepdims=True))
    angles = th.angle(sin_theta[:, 0] + 1j * sin_theta[:, 1])
    indices: list[th.Tensor] = []
    for angle_range in angle_ranges:
        indices.append(
            th.nonzero((angle_range[0] <= angles) * (angles < angle_range[1]))[
                :, 0
            ].tolist()
        )
    return indices


def led_indices_by_radii(
    positions: th.Tensor, radii_ranges: list[th.Tensor]
) -> list[th.Tensor]:
    sin_theta = positions[:, 0:2] / th.sqrt((positions**2).sum(-1, keepdims=True))
    led_na = (sin_theta**2).sum(-1).sqrt()
    indices: list[th.Tensor] = []
    for radius_range in radii_ranges:
        indices.append(
            th.nonzero((radius_range[0] <= led_na) * (led_na < radius_range[1]))[
                :, 0
            ].tolist()
        )

    return indices


def synthetic_led_positions(n_leds: int, pitch: float, z: float):
    led_ra_size = np.floor(n_leds / 2)
    indices = th.arange(-led_ra_size, led_ra_size + 1, device=device)
    led_indices_h, led_indices_v = th.meshgrid(indices, indices, indexing="ij")
    return th.stack(
        (led_indices_v * pitch, led_indices_h * pitch, z * th.ones_like(led_indices_h)),
        dim=-1,
    ).view(n_leds**2, 3)


mps = True
if mps:
    device = th.device("mps")
    dtype = th.float32
    dtypec = th.complex64
else:
    device = th.device("cpu")
    dtype = th.float64
    dtypec = th.complex128


output_root = Path(os.environ["EXPERIMENTS_ROOT"]) / "phaseretrieval" / "simulation"

# in um
led_pitch = 4_000
led_z = 67_500
n_leds = 25
led_positions = synthetic_led_positions(n_leds=n_leds, pitch=led_pitch, z=led_z)

camera_size = 100
lamda = 0.514
na = 0.19
magnification = 8.1485
pixel_size = 5.5
microscope = Microscope(
    led_positions, camera_size, lamda, na, magnification, pixel_size
)

size = 220
shape = (220, 220)
center = (50, 60)
img = skimage.data.camera()
img = (img - np.min(img)) / (np.max(img) - np.min(img)) - 0.5
img = np.exp(1j * img)

v_center = img.shape[0] // 2 - 60
h_center = img.shape[1] // 2 + 50
img = img[
    v_center - size // 2 : v_center + size // 2,
    h_center - size // 2 : h_center + size // 2,
]
image = th.from_numpy(img).to(dtypec).to(device)[None, None]

# DPC
radius_range = [th.Tensor([0, 1]) * na]
angle_ranges = th.Tensor([[0.0, np.pi], [-np.pi / 2, np.pi / 2]])
angle_indices = led_indices_by_angles(led_positions, angle_ranges=angle_ranges)
bf_indices = led_indices_by_radii(led_positions, radius_range)[0]
dpc_indices = [
    th.Tensor(list(set(angle_ind) & set(bf_indices))).to(th.int64)
    for angle_ind in angle_indices
]
model = pp.MultiplexedFourierPtychography(microscope, dpc_indices, shape)
y = model.forward(image)
x_est = pp.DPC(y, model, shape, alpha=0.1)
utils.dump_simulation(th.exp(1j * x_est), image, output_root / "DPC")

# BF-PPR
angle_ranges = th.Tensor([[0.0, np.pi], [-np.pi / 2, np.pi / 1.9], [-np.pi, np.pi]])
angle_indices = led_indices_by_angles(led_positions, angle_ranges=angle_ranges)
bf_ppr_indices = [
    th.Tensor(list(set(angle_ind) & set(bf_indices))).to(th.int64)
    for angle_ind in angle_indices
]
model = pp.MultiplexedFourierPtychography(microscope, bf_ppr_indices, shape)
y = model.forward(image)
x_est = pp.PPR(y, model, shape, alpha=0.1, n_iter=4, inner_iter=100)
utils.dump_simulation(x_est, image, output_root / "BF-PPR")

# DF-PPR-2
radii_ranges = [th.Tensor([na * f1, na * f2]) for (f1, f2) in [(1.0, 2)]]
df_ppr_indices = led_indices_by_radii(led_positions, radii_ranges=radii_ranges)
df_ppr_indices = [th.Tensor(ind).to(th.int64) for ind in df_ppr_indices]
ppr_indices = bf_ppr_indices + df_ppr_indices
model = pp.MultiplexedFourierPtychography(microscope, ppr_indices, shape)
y = model.forward(image)
x_est = pp.PPR(y, model, shape, alpha=0.1, n_iter=30, inner_iter=100)
utils.dump_simulation(x_est, image, output_root / "DF-PPR-2")

# DF-PPR-3
radii_ranges = [
    th.Tensor([na * f1, na * f2]) for (f1, f2) in [(1.0, 1.3), (1.3, 1.7), (1.7, 2.0)]
]
df_ppr_indices = led_indices_by_radii(led_positions, radii_ranges=radii_ranges)
df_ppr_indices = [th.Tensor(ind).to(th.int64) for ind in df_ppr_indices]
ppr_indices = bf_ppr_indices + df_ppr_indices
model = pp.MultiplexedFourierPtychography(microscope, ppr_indices, shape)
y = model.forward(image)
x_est = pp.PPR(y, model, shape, alpha=0.1, n_iter=30, inner_iter=100)
utils.dump_simulation(x_est, image, output_root / "DF-PPR-3")

# FPM
fpm_radius = 2.5 * na
fpm_indices = led_indices_by_radii(led_positions, [th.Tensor([0, fpm_radius])])
fpm_indices = [th.Tensor([index]).to(th.int64) for index in fpm_indices[0]]
model = pp.MultiplexedFourierPtychography(microscope, fpm_indices, shape)
y = model.forward(image)
x_est = pp.FPM(y, model, shape, n_iter=100, tau=4e-2, epsilon=1e-6)
utils.dump_simulation(x_est, image, output_root / "FPM")
