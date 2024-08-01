import utils
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle

#x0 = np.array([2, 1])
#x1 = np.array([1.1, 2.1])
#x1s, x0s = np.meshgrid(x1, x0)
#
#print(x0s)
#print(x1s)
#print("\n")
#
#dx = x1s - x0s
#d2 = np.abs(dx)
#print(d2)
#print("\n")
#i = np.argmin(d2, axis=0)
#print(i)
#print("\n")
#print(np.take_along_axis(d2, i[None], 0))
#print("\n")
#print(np.take_along_axis(x1s, i[None], 0))
#print("\n")
#print(i.shape)
#print("\n")
#print(x1[i])
#print("\n")
#print(d2 >= 0.15)

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

data = pd.read_csv("./clusters.dat", header=None).fillna(-1)
xs = data.iloc[:, 1::2].to_numpy()
ys = data.iloc[:, 2::2].to_numpy()
utils.FixPlot(8, 8)

ind_xs = np.zeros_like(xs)
ind_ys = np.zeros_like(ys)

ind_xs[0, :] = xs[0, :]
ind_ys[0, :] = ys[0, :]

sx = (gi.cols) * gp[0].lattice
sy = (gi.rows) * gp[0].lattice


min_d2s = np.zeros((9, xs.shape[1]))
new_xs = np.zeros((9, xs.shape[1]))
new_ys = np.zeros((9, xs.shape[1]))
invalids = np.zeros((9, xs.shape[1]), dtype=bool)

min_d2s[:, :] = 1e8

for t in range(1, xs.shape[0]):
    def find_min(x0, y0, x1, y1):
        x1s, x0s = np.meshgrid(x1, x0)
        y1s, y0s = np.meshgrid(y1, y0)

        dx = x1s - x0s
        dy = y1s - y0s

        d2 = dx * dx + dy * dy

        i = d2.argmin(axis=0)
        min_d2 = np.take_along_axis(d2, i[None], 0)[0]
        new_x = x1[i]
        new_y = y1[i]
        invalid = ((new_x - x0) ** 2.0 + (new_y - y0) ** 2.0) >= (1.0e-9 ** 2.0)
        return new_x, new_y, min_d2, invalid

    c = 0
    for i in [-1, 0, 1]:
        for j in [-1, 0, 1]:
            new_xs[c, :], new_ys[c, :], min_d2s[c, :], invalids[c, :] = find_min(ind_xs[t - 1, :] + j * sx, ind_ys[t - 1, :] + i * sy, xs[t, :], ys[t, :])
            c += 1

    i = min_d2s.argmin(axis=0)
    min_d2 = np.take_along_axis(min_d2s, i[None], 0)
    new_x = np.take_along_axis(new_xs, i[None], 0)
    new_y = np.take_along_axis(new_ys, i[None], 0)
    invalid = min_d2 >= (1.0e-9 ** 2.0)
    ind_xs[t, :] = new_x
    ind_ys[t, :] = new_y
    ind_xs[t, invalid[0]] = -1

fig, ax = plt.subplots()
ax.set_xlim((0, gi.cols * gp[0].lattice))
ax.set_ylim((0, gi.rows * gp[0].lattice))

with np.printoptions(threshold=np.inf):
    print(ind_xs)
#
#print(xs[0], xs[-1])

plt.subplots_adjust(wspace=0, hspace=0)

c = plt.colormaps["hsv"]
t = data[0] / max(data[0])
ax.scatter(xs, ys, s=0.5)

fig.savefig("./trajectories.png", dpi=cmd.DPI, facecolor="white", bbox_inches="tight")
