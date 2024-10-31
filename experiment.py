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
def PPR(y, microscope, indices, n_iter=100, linear_n_iter=100, alpha=1e5, lamda=0):
    size = max(microscope.reconstruction_size(i) for i in indices)
    shape = (size, size)
    model = phaseretrieval.MultiplexedFourierPtychographyPhaseShift(
        microscope, indices, shape
    )

    # TODO we need to discuss how to properly separate all of this stuff
    # probably have a `residual` function in the model that takes the data as
    # argument. i dont think it makes sense to give the data into the
    # microscope class
    def f(x):
        return y - model.forward(x)

    f.jacobian = model.jacobian

    x0 = th.ones((1, 1, *shape), dtype=dtypec, device=device)
    x0 *= (th.mean(y[0, 0]) / model.forward(x0)[0, 0].mean()).sqrt()

    def cb(x):
        print((f(x) ** 2).sum())
        plt.figure()
        plt.imshow(th.angle(x)[0, 0].cpu().numpy(), cmap='gray')
        plt.show()

    def solve(A, b):
        return algos.conjugate_gradient(
            A + alpha * pl.Id(), b, x0, n_iter=linear_n_iter, dim=(1, 2, 3)
        )

    return algos.gauss_newton(f=f, x0=x0, n_iter=n_iter, solve=solve, callback=cb)


positions = th.from_numpy(np.load('./dome-positions.npy')
                          ).to(dtype).to(device) * 1_000
# Make the positions align with the computation of the shifts and the real
# measurements (switch x and y and multiply them by -1)
positions = th.stack((-positions[:, 1], -positions[:, 0], positions[:, 2])).T
indices = []
with open('./led_indices.txt', 'r') as f:
    while (line := f.readline()) != "":
        indices.append(th.from_numpy(
            np.fromstring(line, sep=',', dtype=np.int32)))

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
print(images_bf.dtype)

images = np.concat((images_bf[:5], images_df[5:] / exposure_df * exposure_bf))[None]
print(images.shape)
h, w = images.shape[2:]
camera_size = 256
ioff = -100
joff = -60
istart = (h - camera_size) // 2 + ioff
jstart = (w - camera_size) // 2 + joff
images = images[:, :, istart:istart + camera_size, jstart:jstart + camera_size]
images = th.from_numpy(images).to(dtype).to(device)
clip = False
if clip:
    for i, im in enumerate(images):
        p = th.quantile(im, 0.99)
        images[i, im > p] = p
images = images[:, 0:]
indices = indices[0:]
print(sum(len(idcs) for idcs in indices))


lamda = 0.525
pixel_size = 6.5
microscope = Microscope(positions, camera_size, lamda,
                        na, magnification, pixel_size)
x_est = PPR(images, microscope, indices)

plt.figure()
plt.imshow(np.fliplr(th.angle(x_est).cpu().numpy()[0, 0]), cmap="gray")
plt.figure()
plt.imshow(th.angle(x_est).cpu().numpy()[0, 0], cmap="gray")
plt.show()
