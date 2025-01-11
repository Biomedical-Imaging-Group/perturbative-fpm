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
    sin_theta = positions[:, 0:2] / th.sqrt((positions**2).sum(-1, keepdim=True))
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
    sin_theta = positions[:, 0:2] / th.sqrt((positions**2).sum(-1, keepdim=True))
    led_na = (sin_theta**2).sum(-1).sqrt()
    indices: list[th.Tensor] = []
    for radius_range in radii_ranges:
        indices.append(
            th.nonzero((radius_range[0] <= led_na) * (led_na <= radius_range[1]))[
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
led_pitch = 2_000
led_z = 67_500
n_leds = 29
led_positions = synthetic_led_positions(n_leds=n_leds, pitch=led_pitch, z=led_z)

camera_size = 100
lamda = 0.514
na = 0.20
magnification = 10
pixel_size = 5
microscope = Microscope(
    led_positions, camera_size, lamda, na, magnification, pixel_size
)

size = 220
shape = (220, 220)
img = skimage.data.camera()
img = ((img - np.min(img)) / (np.max(img) - np.min(img)) - 0.5) * 1
img = np.exp(1j * img)

v_center = 200
h_center = 300
img = img[
    v_center - size // 2 : v_center + size // 2,
    h_center - size // 2 : h_center + size // 2,
]
image = th.from_numpy(img).to(dtypec).to(device)[None, None]
utils.dump_simulation(image, image, output_root / "reference")

# DPC
radius_range = [th.Tensor([0, 1.0]) * na]
angle_ranges = th.Tensor([[0.0, np.pi], [-np.pi / 2, np.pi / 1.9]])
angle_indices = led_indices_by_angles(led_positions, angle_ranges=angle_ranges)
bf_indices = led_indices_by_radii(led_positions, radius_range)[0]
dpc_indices = [
    th.Tensor(list(set(angle_ind) & set(bf_indices))).to(th.int64)
    for angle_ind in angle_indices
]
model = pp.MultiplexedFourierPtychography(microscope, dpc_indices, shape)
y = model.forward(image)
dpc = th.exp(1j * pp.DPC(y, model, shape, alpha=0.1))
utils.draw_patterns(led_positions, na, dpc_indices, output_root / "DPC")
utils.dump_simulation(dpc, image, output_root / "DPC")

# BF-pFPM (prime)
bf_pfpmprime_indices = dpc_indices

# BF-pFPM
angle_ranges = th.Tensor([[0.0, np.pi], [-np.pi / 2, np.pi / 1.9], [-np.pi, np.pi]])
angle_indices = led_indices_by_angles(led_positions, angle_ranges=angle_ranges)
bf_pfpm_indices = [
    th.Tensor(list(set(angle_ind) & set(bf_indices))).to(th.int64)
    for angle_ind in angle_indices
]

# DF-pFPM
radii_ranges = [th.Tensor([na * f1, na * f2]) for (f1, f2) in [(1.0, 1.5), (1.5, 2)]]
df_indices = led_indices_by_radii(led_positions, radii_ranges=radii_ranges)
df_indices = [th.Tensor(ind).to(th.int64) for ind in df_indices]
df_pfpm_indices = bf_pfpm_indices + df_indices

# DF-pFPM (prime)
radii_ranges = [
    th.Tensor([na * f1, na * f2]) for (f1, f2) in [(1.0, 1.3), (1.3, 1.6), (1.6, 2.0)]
]
df_indices = led_indices_by_radii(led_positions, radii_ranges=radii_ranges)
df_indices = [th.Tensor(ind).to(th.int64) for ind in df_indices if len(ind) > 0]
df_pfpmprime_indices = bf_pfpm_indices + df_indices


n_iter = 8
inner_iter = 100

# for indices, name in zip(
#     [bf_pfpmprime_indices, bf_pfpm_indices, df_pfpm_indices, df_pfpmprime_indices],
#     ["BF-pFPMprime", "BF-pFPM", "DF-pFPM", "DF-pFPMprime"],
# ):
#     model = pp.MultiplexedFourierPtychography(microscope, indices, shape)
#     y = model.forward(image)
#     x_est = pp.PPR(y, model, shape, alpha=0.1, n_iter=n_iter, inner_iter=inner_iter)
#     utils.draw_patterns(led_positions, na, indices, output_root / name)
#     utils.dump_simulation(x_est, image, output_root / name)


# FPM
fpm_radius = 2.5 * na
fpm_indices = led_indices_by_radii(led_positions, [th.Tensor([0, fpm_radius])])
fpm_indices = [th.Tensor([index]).to(th.int64) for index in fpm_indices[0]]
# model = pp.MultiplexedFourierPtychography(microscope, fpm_indices, shape)
# y = model.forward(image)
# x_est = pp.FPM(y, model, shape, n_iter=100, tau=4e-2, epsilon=1e-6)
# utils.dump_simulation(x_est, image, output_root / "FPM")


def cone_indices(cone, num_cones):
    radius_range = [th.Tensor([1, 2.0]) * na]
    angle_range = [
        -np.pi + cone * 2 * np.pi / num_cones,
        -np.pi + (cone + 1) * 2 * np.pi / num_cones,
    ]
    angle_ranges = th.Tensor([angle_range])
    angle_indices = led_indices_by_angles(led_positions, angle_ranges=angle_ranges)[0]
    df_indices = led_indices_by_radii(led_positions, radius_range)[0]
    return th.Tensor(list(set(angle_indices) & set(df_indices))).to(th.int64)


# Comparison to mFPM, dark-field cone partitioning from that one learned recovery paper
rng = np.random.default_rng()
for n_patterns in [5, 6]:
    experiment_path = output_root / "pattern-comparison" / str(n_patterns)
    # mFPM with (number of leds) / (number of patterns) turned on per pattern
    # note that this does not guarantee that we turn on all leds as we dont check
    # duplicates between patterns. but no one says that turning all on yields a benefit
    all_indices = led_indices_by_radii(led_positions, [th.Tensor([0, 2 * na])])[0]
    mfpm_indices = []
    for _ in range(n_patterns):
        mfpm_indices.append(
            rng.choice(all_indices, size=n_leds**2 // n_patterns, replace=False)
        )

    # cone-type patterns, dividing the dark field patterns into n_patterns - 2 cones
    # -2 because we already use two bf patterns
    cone_base_indices = bf_pfpm_indices.copy()
    num_cones = n_patterns - 3
    for cone in range(num_cones):
        cone_base_indices.append(cone_indices(cone, num_cones))

    # proposed pattern selection, we already have these indices
    our_indices = df_pfpm_indices if n_patterns == 5 else df_pfpmprime_indices

    # for ind, name in zip(
    #     [mfpm_indices, cone_base_indices, our_indices], ["mFPM", "cone", "pFPM"]
    # ):
    #     model = pp.MultiplexedFourierPtychography(microscope, ind, shape)
    #     y = model.forward(image)
    #     x_est = pp.PPR_PGD(
    #         y, model, shape, alpha=0.1, n_iter=n_iter, inner_iter=inner_iter
    #     )
    #     utils.draw_patterns(led_positions, na, ind, experiment_path / name)
    #     utils.dump_simulation(x_est, image, experiment_path / name)
