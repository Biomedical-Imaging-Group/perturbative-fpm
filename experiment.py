import torch as th
import skimage.transform as sktr
import utils
import numpy as np
import imageio.v3 as imageio
import os
from pathlib import Path
import pyphaseretrieve.phaseretrieval as pp


mps = True
if mps:
    device = th.device("mps")
    dtype = th.float32
    dtypec = th.complex64
else:
    device = th.device("cpu")
    dtype = th.float64
    dtypec = th.complex128

# Get data
base_path = Path(os.environ["DATASETS_ROOT"])
experiment = "attempt_1"
phantom = "usaf"
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
path = (
    base_path
    / "ptycho"
    / experiment
    / f"{phantom}_phase_{exposure_bf}ms_{magnification}x_{na:.2f}NA_20240216"
)
images_bf = imageio.imread(path / "custom_pat.tif").astype(np.float64)

exposure_df = 200
path = (
    base_path
    / "ptycho"
    / experiment
    / f"{phantom}_phase_{exposure_df}ms_{magnification}x_{na:.2f}NA_20240216"
)
images_df = imageio.imread(path / "custom_pat.tif").astype(np.float64)

# Concatenate bf and df iamges and compensate for longer exposure
images = np.concat((images_bf[:5], images_df[5:] / exposure_df * exposure_bf))[None]
h, w = images.shape[2:]
camera_size = 126
# just to get rid of boundary effects in the visualizations, we should check if
# this is even still needed
crop = 8

# Such that we have a nicely centered phantom; up to change
ioff, joff = -90, -10
istart = (h - camera_size) // 2 + ioff
jstart = (w - camera_size) // 2 + joff
images = images[:, :, istart : istart + camera_size, jstart : jstart + camera_size]
images = th.from_numpy(images).to(dtype).to(device)
output_root = Path(os.environ["EXPERIMENTS_ROOT"]) / "phaseretrieval" / "experiments"
# Write central led image to disc for visualizations
brightfield_image = images[0:1].cpu().numpy()[0, 0]
brightfield_image = th.from_numpy(sktr.resize(brightfield_image, shape)[None, None])
dpc_norm = (brightfield_image.mean()*2).sqrt()
utils.dump_experiments(brightfield_image, output_root / "brightfield", crop)

# Setup of physical microscope
lamda = 0.525
pixel_size = 6.5
positions = th.from_numpy(np.load("./dome-positions.npy")).to(dtype).to(device) * 1_000
# Make the positions align with the computation of the shifts and the real
# measurements (switch x and y and multiply them by -1)
positions = th.stack((-positions[:, 1], -positions[:, 0], positions[:, 2])).T
indices = []
with open("./led_indices.txt", "r") as f:
    while (line := f.readline()) != "":
        indices.append(th.from_numpy(np.fromstring(line, sep=",", dtype=np.int64)))

# Microscope is the same for all experiments
microscope = pp.Microscope(positions, camera_size, lamda, na, magnification, pixel_size)

experiments = {
    "BF-pFPMprime": {
        "patterns": [1, 2],
        "n_iter": 4,
        "linear_n_iter": 100,
    },
    "BF-pFPM": {
        "patterns": [0, 1, 2],
        "n_iter": 4,
        "linear_n_iter": 100,
    },
    "DF-pFPM": {
        "patterns": [0, 1, 2, 9, 10],
        "n_iter": 8,
        "linear_n_iter": 100,
    },
    "DF-pFPMprime": {
        "patterns": [0, 1, 2, 6, 7, 8],
        "n_iter": 8,
        "linear_n_iter": 100,
    },
}

for reg, weight in zip(["tv", "l2", "none"], [1.5e4, 9e4, 0]):
    for name, params in experiments.items():
        our_images = images[:, params["patterns"]]
        our_indices = [indices[pattern] for pattern in params["patterns"]]
        model = pp.MultiplexedFourierPtychography(microscope, our_indices, shape)
        x_est = pp.PPR_PGD(
            our_images,
            model,
            shape,
            params["n_iter"],
            params["linear_n_iter"],
            alpha=weight,
            reg=reg,
        )
        utils.dump_experiments(th.angle(x_est), output_root / reg / name, crop)

# DPC experiments
alpha = 5e0
dpc_patterns = [1, 2]
dpc_images = images[:, dpc_patterns]
dpc_indices = [indices[pattern] for pattern in dpc_patterns]
model = pp.MultiplexedFourierPtychography(microscope, dpc_indices, shape)
x_est = pp.DPC(dpc_images, model, shape, alpha)
utils.dump_experiments(x_est / dpc_norm, output_root / "DPC", crop)

# FPM experiments; requires to load the FPM data. For some reason, the x dir
# seems to be flipped in comparison to the prev experiments.. ?
positions = th.from_numpy(np.load("./dome-positions.npy")).to(dtype).to(device) * 1_000
positions = th.stack((positions[:, 1], -positions[:, 0], positions[:, 2])).T
microscope = pp.Microscope(positions, camera_size, lamda, na, magnification, pixel_size)
exposure_fpm = 1000
path = (
    base_path
    / "ptycho"
    / experiment
    / f"{phantom}_target_group5_{exposure_fpm}ms_{magnification}x_{na:.2f}NA_20240216"
)
images = imageio.imread(path / "fpm.tif").astype(np.float64)[None]
images = images[:, :, istart : istart + camera_size, jstart : jstart + camera_size]
images = th.from_numpy(images).to(dtype).to(device)
central_led = images[0, 0].cpu().numpy()
central_led = th.from_numpy(sktr.resize(central_led, shape)[None, None])
utils.dump_experiments(central_led, output_root / "central_led", crop)

# Arbitrary dark-field index
darkfield_led = images[0, 116].cpu().numpy()
darkfield_led = th.from_numpy(sktr.resize(darkfield_led, shape)[None, None])
utils.dump_experiments(darkfield_led, output_root / "darkfield_led", crop)

# FPM indices are just 1...N, leds ordered by NA
indices = th.arange(images.shape[1])[:, None]
model = pp.MultiplexedFourierPtychography(microscope, indices, shape)
tau = 1e-2
x_est = pp.FPM(images, model, shape, n_iter=50, tau=tau, epsilon=1.5e1)
utils.dump_experiments(th.angle(x_est), output_root / "FPM", crop)
