import torch as th
import utils
import numpy as np
import imageio.v3 as imageio
import os
from pathlib import Path
import pyphaseretrieve.phaseretrieval as pp


mps = False
if mps:
    device = th.device("mps")
    dtype = th.float32
    dtypec = th.complex64
else:
    device = th.device("cpu")
    dtype = th.float64
    dtypec = th.complex128

# Get data
base_path = Path(os.environ['DATASETS_ROOT'])
experiment = 'attempt_1'
phantom = 'usaf'
na = 0.25
magnification = 10
# This is tunable in principle but now set to ´max_reconstruction_size´ of any
# of the experiments given the LED positions. I think this is the most elegant
# for comparison, otherwise we need to interpolate between the sizes,
# especially relevant for the simulation (TODO is it actually relevant here...?)
shape = (220, 220)

# We use 50ms exposure for the brightfield images to make sure they are not
# saturated, and 200ms for the darkfield images for higher snr
exposure_bf = 50
path = base_path / 'ptycho' / experiment / \
    f'{phantom}_phase_{exposure_bf}ms_{magnification}x_{na:.2f}NA_20240216'
images_bf = imageio.imread(path / 'custom_pat.tif').astype(np.float64)

exposure_df = 200
path = base_path / 'ptycho' / experiment / \
    f'{phantom}_phase_{exposure_df}ms_{magnification}x_{na:.2f}NA_20240216'
images_df = imageio.imread(path / 'custom_pat.tif').astype(np.float64)

# Concatenate bf and df iamges and compensate for longer exposure
images = np.concat(
    (images_bf[:5], images_df[5:] / exposure_df * exposure_bf))[None]
h, w = images.shape[2:]
camera_size = 126

# Such that we have a nicely centered phantom; up to change
ioff, joff = -90, -10
istart = (h - camera_size) // 2 + ioff
jstart = (w - camera_size) // 2 + joff
images = images[:, :, istart:istart + camera_size, jstart:jstart + camera_size]
images = th.from_numpy(images).to(dtype).to(device)

# Setup of physical microscope
lamda = 0.525
pixel_size = 6.5
positions = th.from_numpy(np.load('./dome-positions.npy')
                          ).to(dtype).to(device) * 1_000
# Make the positions align with the computation of the shifts and the real
# measurements (switch x and y and multiply them by -1)
positions = th.stack((-positions[:, 1], -positions[:, 0], positions[:, 2])).T
indices = []
with open('./led_indices.txt', 'r') as f:
    while (line := f.readline()) != "":
        indices.append(th.from_numpy(
            np.fromstring(line, sep=',', dtype=np.int64)))

# Microscope is the same for all experiments
microscope = pp.Microscope(
    positions, camera_size, lamda, na, magnification, pixel_size
)


# just to get rid of boundary effects in the visualizations, we should check if
# this is even still needed
crop = 8

# "PPR" experiments TODO come up with better name and distinguish them more
# i.e. this "DPC" here is one iteration of GN with the respecitve (nonlinear)
# solver
experiments = {
    'DPC': {
        'patterns': [1, 2],
        'n_iter': 1,
        'linear_n_iter': 100,
    },
    'BF-PPR': {
        'patterns': [0, 1, 2],
        'n_iter': 4,
        'linear_n_iter': 100,
    },
    'DF-PPR-3': {
        'patterns': [0, 1, 2, 6, 7, 8],
        'n_iter': 4,
        'linear_n_iter': 100,
    },
    'DF-PPR-2': {
        'patterns': [0, 1, 2, 9, 10],
        'n_iter': 4,
        'linear_n_iter': 100,
    },
}

output_root = Path(os.environ['EXPERIMENTS_ROOT']) / 'phaseretrieval'
for reg, weight in zip(['tv', 'l2'], [2e5, 1e5]):
    # TODO we can remove this most likely, to flatten the output directory
    # structure we use inexact only anyway
    phase_path = 'inexact-phase'
    for name, params in experiments.items():
        our_images = images[:, params['patterns']]
        our_indices = [indices[pattern] for pattern in params['patterns']]
        model = pp.MultiplexedFourierPtychography(microscope, indices, shape)
        x_est = pp.PPR(our_images, model, params['n_iter'], params['linear_n_iter'], reg=reg)
        utils.dump_experiments(th.angle(x_est), output_root / reg / phase_path / name, crop)


# DPC experiments
alpha = 1e6
dpc_patterns = [1, 2]
dpc_images = images[:, dpc_patterns]
dpc_indices = [indices[pattern] for pattern in dpc_patterns]
model = pp.MultiplexedFourierPtychography(microscope, dpc_indices, shape)
x_est = pp.DPC(dpc_images, model, shape, alpha)
utils.dump_experiments(x_est, output_root / 'DPC', crop)


# FPM experiments; requires to load the FPM data. For some reason, the x dir
# seems to be flipped in comparison to the prev experiments.. ?
positions = th.from_numpy(
    np.load('./dome-positions.npy')
).to(dtype).to(device) * 1_000
positions = th.stack((positions[:, 1], -positions[:, 0], positions[:, 2])).T
microscope = pp.Microscope(
    positions, camera_size, lamda, na, magnification, pixel_size
)
exposure_fpm = 1000
path = base_path / 'ptycho' / experiment / \
    f'{phantom}_target_group5_{exposure_fpm}ms_{
        magnification}x_{na:.2f}NA_20240216'
images = imageio.imread(path / 'fpm.tif').astype(np.float64)[None]
images = images[:, :, istart:istart + camera_size, jstart:jstart + camera_size]
images = th.from_numpy(images).to(dtype).to(device)
# FPM indices are just 1...N, leds ordered by NA
indices = th.arange(images.shape[1])[:, None]
model = pp.MultiplexedFourierPtychography(microscope, indices, shape)
tau = 1e-9
x_est = pp.FPM(images, model, shape, tau=tau)
utils.dump_experiments(th.angle(x_est), output_root / 'FPM', crop)
