import numpy as np
import time
import matplotlib.pyplot as plt
from pyphaseretrieve.phaseretrieval import Microscope
import skimage
import torch as th
import pyphaseretrieve.linop as pl
from pyphaseretrieve import algos, phaseretrieval


def DPC(image, indices: list[th.Tensor], alpha=0):
    # TODO figure out physics and what this is, adapt call signature to what
    # makes sense
    model = phaseretrieval.MultiplexedFourierPtychography(
        microscope, indices, image.shape[2:]
    )
    shape = image.shape[2:]

    # TODO put this factor in the algorithm
    # arises due to normalization of the ffts/iffts of different size
    y = model.forward(image / 2.56)

    crop = pl.Crop2(
        in_shape=shape,
        crop_shape=(camera_size, camera_size),
    )

    numerator = th.zeros(shape, dtype=th.complex64, device=device)[None, None]
    denom = th.zeros(shape, dtype=th.float32, device=device)[None, None]
    probe_ = model.probe.to(th.int32)
    pad = (shape[0] - probe_.shape[2]) // 2
    probe_ = th.fft.fftshift(th.nn.functional.pad(th.fft.ifftshift(probe_[0, 0]), (pad, pad, pad, pad)))[None, None]

    for i_m, shifts in enumerate(model.all_shifts):
        hm = (pl.Roll2(-shifts) @ probe_ - pl.Roll2(shifts) @ probe_).sum(
            1, keepdim=True
        )
        numerator += ((hm * 1j).conj() * pl.Ifftshift() @ crop.T
                      @ pl.Fftshift() @ pl.Fft2() @ y[:, i_m:i_m + 1])
        denom += th.abs(hm) ** 2

    return pl.Ifft2() @ (numerator / (denom + alpha))


# TODO alpha and lamda are unused atm
def PPR(image, indices, n_iter=1, linear_n_iter=1, alpha=0, lamda=0):
    shape = image.shape[2:]
    model = phaseretrieval.MultiplexedFourierPtychography(
        microscope, indices, shape
    )
    y = model.forward(image)

    # TODO we need to discuss how to properly separate all of this stuff
    # probably have a `residual` function in the model that takes the data as
    # argument. i dont think it makes sense to give the data into the
    # microscope class
    def f(x):
        return y - model.forward(x)

    f.jacobian = model.jacobian

    # TODO we could warm start CG with the previous solution if needed
    def solve(A, b, x0):
        return algos.conjugate_gradient(
            A,
            b,
            th.zeros_like(x0),
            n_iter=linear_n_iter,
            dim=(1, 2, 3),
        )

    # TODO remove hardcoded 1, 1
    x0 = th.ones(
        (1, 1, *shape), dtype=th.complex64, device=device)
    start = time.time()
    x_est = algos.gauss_newton(f=f, x0=x0, n_iter=n_iter, solve=solve)
    print(f"PPR time: {time.time() - start}")
    return x_est


# TODO (taken from the phaseretrieval class) the parallelism is extremely
# suboptimal here as it is essentially
# the same code as the multiplexed one which parallelizes over the LEDs
# Here, each forward only has one LED, so we are in the worst case..
def FPM(image, indices: list[th.Tensor], n_iter=5, lr=1):
    shape = image.shape[2:]
    model = phaseretrieval.MultiplexedFourierPtychography(
        microscope, indices, shape
    )
    y = model.forward(image)


    x0 = th.ones((1, 1, *shape), dtype=th.complex64, device=device)

    def nabla(x):
        print(((model.forward(x) - y) ** 2).sum())
        return model.jacobian(x).T @ (model.forward(x) - y)

    return algos.gradient_descent(nabla, lr, x0, n_iter)


def led_indices_by_angles(positions: th.Tensor, angle_ranges: th.Tensor) -> list[th.Tensor]:
    sin_theta = led_positions[:, 0:2] / \
        th.sqrt((led_positions ** 2).sum(-1, keepdims=True))
    angles = th.angle(sin_theta[:, 0] + 1j * sin_theta[:, 1])
    indices: list[th.Tensor] = []
    for i, angle_range in enumerate(angle_ranges):
        indices.append(th.nonzero(
            (angle_range[0] <= angles) * (angles < angle_range[1])
        )[:, 0].tolist())
    return indices


def led_indices_by_radii(positions: th.Tensor, radii_ranges: list[th.Tensor]) -> list[th.Tensor]:
    sin_theta = led_positions[:, 0:2] / \
        th.sqrt((led_positions ** 2).sum(-1, keepdims=True))
    led_na = (sin_theta ** 2).sum(-1).sqrt()
    indices: list[th.Tensor] = []
    for i, radius_range in enumerate(radii_ranges):
        indices.append(th.nonzero(
            (radius_range[0] <= led_na) * (led_na < radius_range[1])
        )[:, 0].tolist())

    return indices


def synthetic_led_positions(n_leds: int, pitch: float, z: float):
    led_ra_size = np.floor(n_leds / 2)
    indices = th.arange(-led_ra_size, led_ra_size + 1, device=device)
    led_indices_h, led_indices_v = th.meshgrid(
        indices, indices, indexing="ij"
    )
    return th.stack((
        led_indices_v * pitch,
        led_indices_h * pitch,
        z * th.ones_like(led_indices_h)),
        dim=-1
    ).view(n_leds ** 2, 3)


if __name__ == "__main__":
    device: th.device = th.device('mps')

    # in um
    led_pitch = 4_000
    led_z = 67_500
    n_leds = 25
    led_positions = synthetic_led_positions(
        n_leds=n_leds, pitch=led_pitch, z=led_z)

    camera_size = 100
    lamda = 0.514
    na = 0.19
    magnification = 8.1485
    pixel_size = 5.5
    microscope = Microscope(led_positions, camera_size,
                            lamda, na, magnification, pixel_size)

    size = 256
    center = (50, 60)
    img = skimage.data.camera()
    img = (img - np.min(img)) / (np.max(img) - np.min(img)) - 0.5
    img = np.exp(1j * img)

    v_center = img.shape[0] // 2 - 60
    h_center = img.shape[1] // 2 + 50
    img = img[
        v_center - size // 2: v_center + size // 2,
        h_center - size // 2: h_center + size // 2,
    ]
    image = th.from_numpy(img).to(th.complex64).to(device)[None, None]

    camera_size = 100
    groundtruth_size = 256

    # TODO these produce the wrong thing for now because of how the angles are
    # constructed
    radius_range = [th.Tensor([0, 1]) * na]
    angle_ranges = th.Tensor([[0.0, np.pi], [-np.pi / 2, np.pi / 2]])

    # TODO make this cleaner, although it doesnt really matter as its only
    # the simulation...
    angle_indices = led_indices_by_angles(
        led_positions, angle_ranges=angle_ranges)
    bf_indices = led_indices_by_radii(led_positions, radius_range)[0]
    # Get only indices that are both in the angle wedge and the bright field
    dpc_indices = [th.Tensor(list(set(angle_ind) & set(bf_indices)))
                   for angle_ind in angle_indices]
    x_est = DPC(image, dpc_indices, alpha=0.1)
    plt.figure()
    plt.imshow(x_est.real.cpu().numpy()[0, 0])
    plt.colorbar()

    # TODO this angle business needs to be cleaned
    # with the following, i can reproduce the patterns from the student
    angle_ranges = th.Tensor(
        [[0., np.pi], [-np.pi / 2, np.pi / 1.9], [-np.pi, np.pi]]
    )
    angle_indices = led_indices_by_angles(
        led_positions, angle_ranges=angle_ranges)
    bf_ppr_indices = [th.Tensor(list(set(angle_ind) & set(bf_indices)))
                      for angle_ind in angle_indices]
    x_est_bf = PPR(image, bf_ppr_indices, n_iter=2, linear_n_iter=10)
    plt.figure()
    plt.imshow(th.angle(x_est_bf).cpu().numpy()[0, 0])
    plt.colorbar()

    radii_ranges = [
        th.Tensor([na * f1, na * f2]) for (f1, f2) in [(1., 1.5), (1.5, 2.)]
    ]

    df_ppr_indices = led_indices_by_radii(
        led_positions, radii_ranges=radii_ranges)
    df_ppr_indices = [th.Tensor(ind) for ind in df_ppr_indices]
    ppr_indices = bf_ppr_indices + df_ppr_indices
    x_est_df = PPR(image, ppr_indices, n_iter=2, linear_n_iter=10)
    plt.figure()
    plt.imshow(th.angle(x_est_df).cpu().numpy()[0, 0])
    plt.colorbar()

    print("FPM Start \n----------------------")
    fpm_radius = 2.5 * na
    fpm_indices = led_indices_by_radii(led_positions, [th.Tensor([0, fpm_radius])])
    fpm_indices = [th.Tensor([index]) for index in fpm_indices[0]]
    print(len(fpm_indices))

    x_est = FPM(image, fpm_indices, n_iter=200, lr=1e-4)
    plt.figure()
    plt.imshow(th.angle(x_est).cpu().numpy()[0, 0])
    plt.colorbar()
    plt.show()
