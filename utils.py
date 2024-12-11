import numpy as np
import torch as th
import imageio.v3 as imageio
import os
from pathlib import Path


def dump_experiments(x: th.Tensor, path: Path, crop: int):
    lineplot_j = 196
    lineplot_istart = 115
    lineplot_iend = 128
    plot_xaxis = np.arange(lineplot_iend - lineplot_istart)
    x = np.fliplr(x[0, 0, crop:-crop, crop:-crop].cpu().numpy())

    if not os.path.exists(path):
        os.makedirs(path)

    vals = x[lineplot_istart:lineplot_iend, lineplot_j]
    vals = np.stack((plot_xaxis, vals)).T
    np.savetxt(
        path / "vals.csv", vals, delimiter=",", header="x,y", comments="", fmt="%.5f"
    )

    span = np.array([x.min().item(), x.max().item()])[None]
    np.savetxt(path / "span.csv", span, delimiter=",", fmt="%.1f")

    x -= x.min()
    x /= x.max()
    x *= 255.0
    x = x.astype(np.uint8)
    imageio.imwrite(path / "x_est.png", x)


def snr(x, y):
    return 10 * th.log10((x**2).sum() / ((x - y) ** 2).sum())


def rmse(x, y):
    return ((x - y) ** 2).mean().sqrt()


def dump_simulation(x, ref, path):
    metrics = np.array([snr(th.angle(x), th.angle(ref)).cpu().numpy(), rmse(th.angle(x), th.angle(ref)).cpu().numpy()])[None]
    np.savetxt(path / "metrics.csv", metrics, delimiter=",", fmt="%.2f")
    ref_ft = th.fft.fft2(th.angle(ref))
    x_ft = th.fft.fft2(th.angle(x))
    ft_error = th.fft.fftshift(th.abs(ref_ft - x_ft) / th.abs(ref_ft)).clamp_max(1)

    if not os.path.exists(path):
        os.makedirs(path)

    x = th.angle(x)
    span = np.array([x.min().item().real, x.max().item().real])[None]
    np.savetxt(path / "span.csv", span, delimiter=",", fmt="%.1f")
    x = x.cpu().numpy().squeeze()
    x -= x.min()
    x /= x.max()
    x *= 255.0
    x = x.astype(np.uint8)
    imageio.imwrite(path / "x_est.png", x)

    ft_error = ft_error.cpu().numpy().squeeze()
    ft_error -= ft_error.min()
    ft_error /= ft_error.max()
    ft_error *= 255.0
    ft_error = ft_error.astype(np.uint8)
    imageio.imwrite(path / "ft_error.png", ft_error)
