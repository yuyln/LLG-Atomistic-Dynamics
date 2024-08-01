import utils
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle

cmd = utils.CMDArgs("dummy", "dummy")
frames, gi, gp, raw = utils.ReadAnimationBinary("integrate_evolution.dat")

data = pd.read_csv("./clusters_org.dat", header=None).fillna(-1)
xs = data.iloc[:, 1::2].to_numpy()
ys = data.iloc[:, 2::2].to_numpy()
utils.FixPlot(8, 8)

fig, ax = plt.subplots()
ax.set_xlim((0, gi.cols * gp[0].lattice))
ax.set_ylim((0, gi.rows * gp[0].lattice))

ax.set_xticklabels((f"{i/1e-9:.0f}" for i in ax.get_xticks()))
ax.set_yticklabels((f"{i/1e-9:.0f}" for i in ax.get_yticks()))
ax.set_xlabel("$x$(nm)")
ax.set_ylabel("$y$(nm)")

plt.subplots_adjust(wspace=0, hspace=0)

for k in range(xs.shape[1]):
    ax.scatter(xs[:, k], ys[:, k], s=0.5)

fig.savefig("./trajectories.png", dpi=cmd.DPI, facecolor="white", bbox_inches="tight")
