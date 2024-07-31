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

min_d2s = np.zeros((3, 3))
sx = (gi.cols - 1) * gp[0].lattice
sy = (gi.rows - 1) * gp[0].lattice


for t in range(1, xs.shape[0]):
    invalids = np.zeros((3, 3, xs.shape[1]), dtype=bool)
    min_d2s = np.zeros((3, 3, xs.shape[1]))
    sdxs = np.zeros((3, 3, xs.shape[1]))
    sdys = np.zeros((3, 3, xs.shape[1]))

    def find_min(x0, y0, x1, y1):
        x1s, x0s = np.meshgrid(x1, x0)
        y1s, y0s = np.meshgrid(y1, y0)

        dx = x1s - x0s
        dy = y1s - y0s

        d2 = dx * dx + dy * dy

        i = d2.argmin(axis=0)
        j = d2.argmin(axis=1)

        min_d2 = d2[i, j]

        sdx = dx[i, j]
        sdy = dy[i, j]

        invalid = min_d2 >= 2.0e-9 ** 2.0
        return invalid, sdx, sdy, min_d2

    for i in [-1, 0, 1]:
        for j in [-1, 0, 1]:
            invalids[i + 1, j + 1, :], sdxs[i + 1, j + 1, :], sdys[i + 1, j + 1, :], min_d2s[i + 1, j + 1, :] = find_min(xs[t - 1, :] + i * sx, ys[t - 1, :] + i * sy, xs[t, :], ys[t, :])

    invalids = invalids.reshape((9, -1))
    min_d2s = min_d2s.reshape((9, -1))
    sdxs = sdxs.reshape((9, -1))
    sdys = sdys.reshape((9, -1))
    min_i = min_d2s.argmin(axis=0)
    min_j = min_d2s[min_i].argmin(axis=1)


    ind_xs[t, :] = ind_xs[t - 1, :] + sdxs[min_i, min_j]
    ind_ys[t, :] = ind_xs[t - 1, :] + sdys[min_i, min_j]

    ind_xs[t, invalids[min_i, min_j]] = -1
    ind_ys[t, invalids[min_i, min_j]] = -1
    continue

    idxs = []
    for i in range(xs.shape[1]):
        min_d2 = 1e9
        min_idx = -1
        for j in range(xs.shape[1]):
            if xs[t - 1, j] == np.nan:
                continue
            for yy in [-1, 0, 1]:
                for xx in [-1, 0, 1]:
                    dx = xs[t, j] - ind_xs[t - 1][i] + (gi.cols - 1) * gp[0].lattice * xx
                    dy = ys[t, j] - ind_ys[t - 1][i] + (gi.rows - 1) * gp[0].lattice * yy
                    d2 = dx * dx + dy * dy
                    if d2 >= 2e-9 * 2e-9:
                        continue
                    if d2 <= min_d2:
                        min_d2 = d2
                        min_idx = j

        idxs.append(min_idx)

    ind_xs.append([0 for i in range(xs.shape[1])])
    ind_ys.append([0 for i in range(xs.shape[1])])

    for i in range(xs.shape[1]):
        if idxs[i] >= 0:
            ind_xs[-1][i] = xs[t, idxs[i]]
            ind_ys[-1][i] = ys[t, idxs[i]]
        else:
            ind_xs[-1][i] = -1
            ind_ys[-1][i] = -1

            
#for i in range(len(ind_xs)):
#    ind_xs[i] = np.array(ind_xs[i])
#    ind_ys[i] = np.array(ind_ys[i])
#ind_xs = np.array(ind_xs)
#ind_ys = np.array(ind_ys)

fig, ax = plt.subplots(ncols=3)

for i in range(3):
    ax[i].set_xlim((0, (gi.cols - 1) * gp[0].lattice))
    ax[i].set_ylim((0, (gi.rows - 1) * gp[0].lattice))
    ax[i].set_xticks(())
    ax[i].set_yticks(())

print(ind_xs)

plt.subplots_adjust(wspace=0, hspace=0)

print(xs.max(), (gi.cols - 1) * gp[0].lattice, gi.cols * gp[0].lattice)
for i in range(3):
    ax[i].scatter(xs, ys, s=1)
    ax[i].scatter(xs / (gi.cols * gp[0].lattice) * (gi.cols - 1) * gp[0].lattice, ys / (gi.rows * gp[0].lattice) * (gi.rows - 1) * gp[0].lattice, s=0.5)
#for i in range(xs.shape[1]):
#    for j in range(3):
#        ax[j].scatter(ind_xs[:, i], ind_ys[:, i], s=0.5)
fig.savefig("./trajectories.png", dpi=cmd.DPI, facecolor="white", bbox_inches="tight")
