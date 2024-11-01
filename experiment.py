import torch as th
import matplotlib.pyplot as plt
import numpy as np
import imageio.v3 as imageio
import os
from pathlib import Path
from pyphaseretrieve.phaseretrieval import Microscope
import pyphaseretrieve.phaseretrieval as phaseretrieval
import pyphaseretrieve.linop as pl
import pyphaseretrieve.algos as algos

mps = True
if mps:
    device = th.device("mps")
    dtype = th.float32
    dtypec = th.complex64
else:
    device = th.device("cpu")
    dtype = th.float64
    dtypec = th.complex128


# TODO refactor such that same code is called from experiment and simulation
def PPR(y, microscope, indices, n_iter=4, linear_n_iter=100, alpha=1e5, exact_phase=True):
    size = max(microscope.reconstruction_size(i) for i in indices)
    shape = (size, size)
    shape = (220, 220)  # TODO this is hardcoded for the experiments ATM, REMOVE!!
    # TODO the results with inexact phase look better; not sure whats going on
    model = phaseretrieval.MultiplexedFourierPtychography(
        microscope, indices, shape, exact_phase=exact_phase
    )

    def f(x):
        return y - model.forward(x)

    f.jacobian = model.jacobian

    x0 = th.ones((1, 1, *shape), dtype=dtypec, device=device)
    x0 *= (th.mean(y[0, 0]) / model.forward(x0)[0, 0].mean()).sqrt()

    def solve(A, b):
        return algos.conjugate_gradient(
            A + alpha * pl.Id(), b, x0, n_iter=linear_n_iter, dim=(1, 2, 3)
        )

    return algos.gauss_newton(f=f, x0=x0, n_iter=n_iter, solve=solve)


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

base_path = Path(os.environ['DATASETS_ROOT'])
experiment = 'attempt_1'
phantom = 'usaf'
exposure_bf = 50
na = 0.25
magnification = 10
path = base_path / 'ptycho' / experiment / \
    f'{phantom}_phase_{exposure_bf}ms_{magnification}x_{na:.2f}NA_20240216'
images_bf = imageio.imread(path / 'custom_pat.tif').astype(np.float64)

exposure_df = 200
path = base_path / 'ptycho' / experiment / \
    f'{phantom}_phase_{exposure_df}ms_{magnification}x_{na:.2f}NA_20240216'
images_df = imageio.imread(path / 'custom_pat.tif').astype(np.float64)

images = np.concat(
    (images_bf[:5], images_df[5:] / exposure_df * exposure_bf))[None]
h, w = images.shape[2:]
camera_size = 128
ioff, joff = -90, -10
istart = (h - camera_size) // 2 + ioff
jstart = (w - camera_size) // 2 + joff
images = images[:, :, istart:istart + camera_size, jstart:jstart + camera_size]
images = th.from_numpy(images).to(dtype).to(device)

clip = False
if clip:
    for i, im in enumerate(images):
        p = th.quantile(im, 0.99)
        images[i, im > p] = p


lamda = 0.525
pixel_size = 6.5
microscope = Microscope(positions, camera_size, lamda,
                        na, magnification, pixel_size)

experiments = {
    'DPC': {
        'patterns': [1, 2],  # TODO use 1&2 or 3&4?
        'n_iter': 1,
        'linear_n_iter': 200,
    },
    'BF-PPR': {
        'patterns': [0, 1, 2],
        'n_iter': 4,
        'linear_n_iter': 100,
    },
    'DF-PPR-3': {
        'patterns': [0, 1, 2, 6, 7, 8],  # TODO ask jonathan if 5 or 6
        'n_iter': 4,
        'linear_n_iter': 100,
    },
    'DF-PPR-2': {
        'patterns': [0, 1, 2, 9, 10],
        'n_iter': 4,
        'linear_n_iter': 100,
    },
}

crop = 8

output_root = Path(os.environ['EXPERIMENTS_ROOT']) / 'phaseretrieval'
for exact_phase in [True, False]:
    phase_path = 'exact-phase' if exact_phase else 'inexact-phase'
    for name, params in experiments.items():
        our_images = images[:, params['patterns']]
        our_indices = [indices[pattern] for pattern in params['patterns']]
        x_est = PPR(our_images, microscope, our_indices,
                    params['n_iter'], params['linear_n_iter'], exact_phase=exact_phase)
        x_est = th.angle(x_est[0, 0, crop:-crop, crop:-crop])
        x_est -= x_est.min()
        x_est /= x_est.max()
        x_est *= 255.
        x_est = x_est.cpu().numpy().astype(np.uint8)

        output_path = output_root / phase_path / name
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        imageio.imwrite(output_path / 'x_est.png', np.fliplr(x_est))
