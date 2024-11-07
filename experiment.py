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


lineplot_j = 196
lineplot_istart = 115
lineplot_iend = 128
plot_xaxis = np.arange(lineplot_iend - lineplot_istart)


# TODO refactor such that same code is called from experiment and simulation
def PPR(y, microscope, indices, n_iter=4, linear_n_iter=100, alpha=1e5, reg='l2', exact_phase=True):
    size = max(microscope.reconstruction_size(i) for i in indices)
    shape = (size, size)
    # TODO this is hardcoded for the experiments ATM, REMOVE!!
    shape = (220, 220)
    # TODO the results with inexact phase look better; not sure whats going on
    model = phaseretrieval.MultiplexedFourierPtychography(
        microscope, indices, shape, exact_phase=exact_phase
    )

    def f(x):
        return y - model.forward(x)

    f.jacobian = model.jacobian

    x0 = th.ones((1, 1, *shape), dtype=dtypec, device=device)
    x0 *= (th.mean(y[0, 0]) / model.forward(x0)[0, 0].mean()).sqrt()

    lamda = 2e5

    def solve(J, x):
        if reg == 'tv':
            nonlocal lamda
            lamda /= 2

            b = algos.power_iteration(J.T @ J, x, n_iter=10)
            opnormJTJ = (b * ((J.T @ J) @ b)).real.sum() / (b * b).real.sum()
            opnormD = np.sqrt(8)
            sigma = 1 / opnormD
            fac = th.sqrt(opnormJTJ)
            sigma *= fac
            sigmaLsqlH = sigma * opnormD ** 2 + opnormJTJ
            tau = 1 / sigmaLsqlH

            def nabla_h(x_):
                return J.T @ (J @ x_ + model.forward(x) - y)

            def prox_g(x):
                return x

            def prox_fs(y):
                y_ = y + lamda * sigma * pl.Grad() @ x
                return y_ / th.maximum(
                    (y_.abs() ** 2).sum(1, keepdims=True).sqrt() / lamda,
                    th.ones(y.shape, device=y.device)
                )

            return algos.condat_vu(pl.Grad(), prox_g, prox_fs, nabla_h, tau, sigma, x, pl.Grad() @ x, n_iter=100)
        else:
            return algos.conjugate_gradient(J.T @ J + alpha * pl.Id(), J.T @ f(x), th.zeros_like(x), n_iter=linear_n_iter)

    return algos.irgn(f=f, x0=x0, n_iter=n_iter, solve=solve)


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
camera_size = 126
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
        'linear_n_iter': 100,
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
for reg in ['tv', 'l2']:
    print(reg)
    for exact_phase in [False]:
        phase_path = 'exact-phase' if exact_phase else 'inexact-phase'
        for name, params in experiments.items():
            print(name)
            our_images = images[:, params['patterns']]
            our_indices = [indices[pattern] for pattern in params['patterns']]
            x_est = PPR(our_images, microscope, our_indices,
                        params['n_iter'], params['linear_n_iter'], exact_phase=exact_phase, reg=reg)
            x_est = np.fliplr(th.angle(x_est[0, 0, crop:-crop, crop:-crop]).cpu().numpy())

            output_path = output_root / reg / phase_path / name
            if not os.path.exists(output_path):
                os.makedirs(output_path)

            vals = x_est[lineplot_istart:lineplot_iend, lineplot_j]
            vals = np.stack((plot_xaxis, vals)).T
            np.savetxt(output_path / 'vals.csv', vals,
                       delimiter=",", header='x,y', comments='', fmt='%.5f')

            span = np.array([x_est.min().item(), x_est.max().item()])[None]
            np.savetxt(output_path / 'span.csv', span,
                       delimiter=",", fmt='%.1f')

            x_est -= x_est.min()
            x_est /= x_est.max()
            x_est *= 255.
            x_est = x_est.astype(np.uint8)
            imageio.imwrite(output_path / 'x_est.png', x_est)


def DPC(y, indices: list[th.Tensor], alpha=5e1) -> th.Tensor:
    # TODO figure out physics and what this is, adapt call signature to what
    # makes sense
    shape = (220, 220)
    model = phaseretrieval.MultiplexedFourierPtychography(
        microscope, indices, shape, exact_phase=False,
    )

    crop = pl.Crop2(
        in_shape=shape,
        crop_shape=(camera_size, camera_size),
    )

    numerator = th.zeros(shape, dtype=th.complex64, device=device)[None, None]
    denom = th.zeros(shape, dtype=th.float32, device=device)[None, None]
    probe_ = model.probe.to(th.int32)
    pad = (shape[0] - probe_.shape[2]) // 2
    probe_ = th.fft.fftshift(th.nn.functional.pad(th.fft.ifftshift(probe_[0, 0]), (pad, pad, pad, pad)))[None, None]
    all_shifts = [th.round(microscope.shifts[index]).to(th.int64) for index in indices]

    for i_m, shifts in enumerate(all_shifts):
        hm = (pl.Roll2(-shifts) @ probe_ - pl.Roll2(shifts) @ probe_).sum(
            1, keepdim=True
        )
        numerator += ((hm * 1j).conj() * pl.Ifftshift() @ crop.T
                      @ pl.Fftshift() @ pl.Fft2() @ y[:, i_m:i_m + 1])
        denom += th.abs(hm) ** 2

    return (pl.Ifft2() @ (numerator / (denom + alpha))).real

dpc_patterns = [1, 2]
dpc_images = images[:, dpc_patterns]
dpc_indices = [indices[pattern] for pattern in dpc_patterns]
x_est = DPC(dpc_images, dpc_indices)
x_est = np.fliplr(x_est[0, 0, crop:-crop, crop:-crop].cpu().numpy())
output_path = output_root / 'DPC'
if not os.path.exists(output_path):
    os.makedirs(output_path)

vals = x_est[lineplot_istart:lineplot_iend, lineplot_j]
vals = np.stack((plot_xaxis, vals)).T
np.savetxt(output_path / 'vals.csv', vals,
           delimiter=",", header='x,y', comments='', fmt='%.5f')
span = np.array([x_est.min().item().real, x_est.max().real.item()])[None]
np.savetxt(output_path / 'span.csv', span, delimiter=",", fmt='%.3f')
x_est -= x_est.min()
x_est /= x_est.max()
x_est *= 255.
x_est = x_est.astype(np.uint8)
imageio.imwrite(output_path / 'x_est.png', x_est)

positions = th.from_numpy(np.load('./dome-positions.npy')
                          ).to(dtype).to(device) * 1_000
# TODO why is this xy plane defined differently here..?
positions = th.stack((positions[:, 1], -positions[:, 0], positions[:, 2])).T
microscope = Microscope(positions, camera_size, lamda,
                        na, magnification, pixel_size)


def FPM(y, indices: list[th.Tensor], n_iter=50, lr=1e-6):
    shape = (220, 220)  # TODO remove hardcoded
    model = phaseretrieval.MultiplexedFourierPtychography(
        microscope, indices, shape, exact_phase=False
    )

    x0 = th.ones((1, 1, *shape), dtype=th.complex64, device=device)
    x0 *= (th.mean(y[0, 0]) / model.forward(x0)[0, 0].mean()).sqrt()

    def nabla(x):
        return model.jacobian(x).T @ (model.forward(x) - y)

    return algos.gradient_descent(nabla, lr, x0, n_iter)


exposure_fpm = 1000
path = base_path / 'ptycho' / experiment / \
    f'{phantom}_target_group5_{exposure_fpm}ms_{
        magnification}x_{na:.2f}NA_20240216'
images = imageio.imread(path / 'fpm.tif').astype(np.float64)[None]
images = images[:, :, istart:istart + camera_size, jstart:jstart + camera_size]
images = th.from_numpy(images).to(dtype).to(device)
indices = th.arange(images.shape[1])[:, None]
x_est = FPM(images, indices)

x_est = np.fliplr(th.angle(x_est[0, 0, crop:-crop, crop:-crop]).cpu().numpy())
output_path = output_root / 'FPM'
if not os.path.exists(output_path):
    os.makedirs(output_path)

vals = x_est[lineplot_istart:lineplot_iend, lineplot_j]
vals = np.stack((plot_xaxis, vals)).T
np.savetxt(output_path / 'vals.csv', vals,
           delimiter=",", header='x,y', comments='', fmt='%.5f')
span = np.array([x_est.min().item(), x_est.max().item()])[None]
np.savetxt(output_path / 'span.csv', span, delimiter=",", fmt='%.3f')
x_est -= x_est.min()
x_est /= x_est.max()
x_est *= 255.
x_est = x_est.astype(np.uint8)
imageio.imwrite(output_path / 'x_est.png', x_est)
