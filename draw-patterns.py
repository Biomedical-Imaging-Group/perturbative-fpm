import torch as th
import os
import math
import matplotlib.pyplot as plt
import simulation
from matplotlib.patches import Circle


def draw_patterns(indices, name):
    for i, pat_indices in enumerate(indices):
        fig, ax = plt.subplots(frameon="false", figsize=(5, 5))
        fig.patch.set_facecolor("k")
        r = (
            math.sqrt(simulation.na**2 * simulation.led_z**2 / (1 - simulation.na**2))
            / 1000
        )
        circle = Circle((0, 0), r, facecolor="gray", edgecolor="none", linewidth=None)
        ax.patch.set_facecolor("k")
        ax.axis("equal")
        ax.set_axis_off()
        ax.add_patch(circle)
        ax.scatter(
            *positions[:, :2].T.cpu().numpy() / 1000,
            facecolors="none",
            edgecolors="w",
            s=100,
        )
        for idx in pat_indices:
            ax.scatter(
                *positions[idx, :2].cpu().numpy() / 1000,
                facecolors="w",
                edgecolors="w",
                s=100,
            )

        path = simulation.output_root / name / "patterns"
        if not os.path.exists(path):
            os.makedirs(path)
        fig.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
        fig.savefig(path / f"{i:02d}.pdf")
        plt.close()


positions = simulation.led_positions
sin_theta = positions[:, 0:2] / th.sqrt((positions**2).sum(-1, keepdim=True))
led_na = (sin_theta**2).sum(-1).sqrt()
na_cutoff = simulation.na
for name, indices in zip(
    ["DPC", "BF-PPR", "DF-PPR-two", "DF-PPR-many"],
    [
        simulation.dpc_indices,
        simulation.bf_ppr_indices,
        simulation.ppr_indices_two,
        simulation.ppr_indices_many,
    ],
):
    print(indices)
    draw_patterns(indices, name)
