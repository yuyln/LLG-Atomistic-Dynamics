import utils
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle

cmd = utils.CMDArgs("dummy", "dummy")
frames, gi, gp, raw = utils.ReadAnimationBinary("integrate_evolution.dat")
if 0:
    utils.FixPlot(8, 8)
    
    data = pd.read_csv("testing.dat", header=None)
    
    fig, ax = plt.subplots()
    #plt.plot(data[0], data[3], "-o", lw=2)
    #plt.show()
    
    ii = data[0].to_numpy()
    for n, i in enumerate(ii):
        y = 31 + 1.5 * i
        for j in range(10):
            x = i + j
            x = ((x % 64) + 64) % 64
            rect = plt.Rectangle((x - 0.5, y - 0.5), 1, 1, facecolor="#00000090")
            ax.add_patch(rect)
            x = i + j
            rect = plt.Rectangle((x - 0.5, y - 0.5), 1, 1, facecolor="#00000090")
            ax.add_patch(rect)
    
        correct = i + 4.5
        correct = correct - np.floor(correct / 64) * 64
        print(correct)
        ax.scatter([data[3][n]], [y], color="red", s=50)
        ax.scatter([correct], [y], color="green", s=35)
    
    
    ax.axvline(63 + 0.5)
    ax.axvline(0 - 0.5)
    
    ax.set_xlim((0 - 0.5, 63 + 0.5))
    ax.set_ylim((0 - 0.5, 63 + 0.5))
    
    plt.show()
    
    exit(1)

data = pd.read_csv("./clusters.dat", header=None)
xs = data.iloc[:, 1::2].to_numpy()
ys = data.iloc[:, 2::2].to_numpy()
utils.FixPlot(8, 8)

ind_xs = np.zeros_like(xs)
ind_ys = np.zeros_like(ys)

ind_xs[0, :] = xs[0, :]
ind_ys[0, :] = ys[0, :]

sx = (gi.cols) * gp[0].lattice
sy = (gi.rows) * gp[0].lattice


invalids = np.zeros((9, xs.shape[1]), dtype=bool)
min_d2s = np.zeros((9, xs.shape[1]))
sdxs = np.zeros((9, xs.shape[1]))
sdys = np.zeros((9, xs.shape[1]))

for t in range(1, xs.shape[0]):
    def find_min(x0, y0, x1, y1):
        x1s, x0s = np.meshgrid(x1, x0)
        y1s, y0s = np.meshgrid(y1, y0)

        dx = x1s - x0s
        dy = y1s - y0s

        d2 = dx * dx + dy * dy

        i = d2.argmin(axis=0)
        idxs = np.indices(i.shape)

        min_d2 = d2[i, idxs][0]
        sdx = dx[i, idxs][0]
        sdy = dy[i, idxs][0]

        invalid = min_d2 >= (5e-9 ** 2.0)
        return invalid, sdx, sdy, min_d2

    c = 0
    for i in [-1, 0, 1]:
        for j in [-1, 0, 1]:
            invalids[c, :], sdxs[c, :], sdys[c, :], min_d2s[c, :] = find_min(xs[t - 1, :] + j * sx, ys[t - 1, :] + i * sy, xs[t, :], ys[t, :])
            c += 1

    i = min_d2s.argmin(axis=0)
    idxs = np.indices(i.shape)

    ind_xs[t, :] = ind_xs[t - 1, :] + sdxs[i, idxs][0]
    ind_ys[t, :] = ind_ys[t - 1, :] + sdys[i, idxs][0]

    ind_xs[t, invalids[i, idxs][0]] = -1
    ind_ys[t, invalids[i, idxs][0]] = -1

ind_xs = ind_xs - np.floor(ind_xs / sx) * sx
ind_ys = ind_ys - np.floor(ind_ys / sy) * sy
fig, ax = plt.subplots(ncols=3, nrows=3)

for i in range(3):
    for j in range(3):
        ax[i][j].set_xlim((0, (gi.cols - 1) * gp[0].lattice))
        ax[i][j].set_ylim((0, (gi.rows - 1) * gp[0].lattice))
        ax[i][j].set_xticks(())
        ax[i][j].set_yticks(())

print(ind_xs)

plt.subplots_adjust(wspace=0, hspace=0)

print(xs.max(), (gi.cols - 1) * gp[0].lattice, gi.cols * gp[0].lattice)
for i in range(3):
    for j in range(3):
        for k in range(xs.shape[1]):
            ax[i][j].scatter(ind_xs[:, k] / (gi.cols * gp[0].lattice) * (gi.cols - 1) * gp[0].lattice, ind_ys[:, k] / (gi.rows * gp[0].lattice) * (gi.rows - 1) * gp[0].lattice, s=0.5)

print(ax[0][0].get_xlim())
print(ax[0][0].get_ylim())
fig.savefig("./trajectories.png", dpi=cmd.DPI, facecolor="white", bbox_inches="tight")
