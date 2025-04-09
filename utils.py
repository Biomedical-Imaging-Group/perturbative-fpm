import numpy as np
import torch as th
import imageio.v3 as imageio
import os
from pathlib import Path
import math
import matplotlib.pyplot as plt
from matplotlib.patches import Circle


green = [45, 96, 54]
green = np.array(green) / 255.0 * 1.6
bfcolor = [45, 93, 123]
bfcolor = np.array(bfcolor) / 255.0 * 1.2
dfcolor = bfcolor * 0.5
# dfcolor = [46 / 255., 40 / 255., 42 / 255.]


def draw_patterns(
    positions: th.Tensor, na: float, indices: list[th.Tensor], root_path: Path
):
    for i, pat_indices in enumerate(indices):
        fig, ax = plt.subplots(frameon="false", figsize=(5, 5))
        r = math.sqrt(na**2 * positions[0, 2] ** 2 / (1 - na**2)) / 1000
        r_df = (
            math.sqrt((2 * na) ** 2 * positions[0, 2] ** 2 / (1 - (2 * na) ** 2)) / 1000
        )
        circle_df = Circle(
            (0, 0), r_df, facecolor=dfcolor, edgecolor="none", linewidth=None
        )
        circle_bf = Circle(
            (0, 0), r, facecolor=bfcolor, edgecolor="none", linewidth=None
        )
        ax.axis("equal")
        ax.set_axis_off()
        ax.add_patch(circle_df)
        ax.add_patch(circle_bf)
        for idx in pat_indices:
            ax.scatter(
                *positions[idx, :2].cpu().numpy() / 1000,
                c=[green],
                s=50,
            )

        path = root_path / "patterns"
        if not os.path.exists(path):
            os.makedirs(path)
        fig.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
        fig.savefig(path / f"{i:02d}.pdf", transparent=True)
        plt.close()


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
    if not os.path.exists(path):
        os.makedirs(path)

    metrics = np.array(
        [
            snr(th.angle(ref), th.angle(x)).cpu().numpy(),
            rmse(th.angle(x), th.angle(ref)).cpu().numpy(),
        ]
    )[None]
    np.savetxt(path / "metrics.csv", metrics, delimiter=",", fmt="%.2f")
    ref_ft = th.fft.fft2(th.angle(ref))
    x_ft = th.fft.fft2(th.angle(x))
    ft_error = th.fft.fftshift(th.abs(ref_ft - x_ft) / th.abs(ref_ft + 1e-6)).clamp_max(
        1
    )

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
