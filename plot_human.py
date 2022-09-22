from Plooter import *
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from matplotlib import cm
from matplotlib.colors import LinearSegmentedColormap

reduce_fac = 2
data = pd.read_table("./output/end.out", header=None)
#data = pd.read_table("./input/starting.in", header=None)
try:
    ani = pd.read_table("./input/anisotropy.in", header=None, delimiter=" ", skiprows=2)
    row_ani = ani[0] + 0.5
    col_ani = ani[1] + 0.5
except:
    row_ani = []
    col_ani = []

try:
    pin = pd.read_table("./input/pinning.in", header=None, delimiter=" ", skiprows=2)
    row_pin = pin[0] + 0.5
    col_pin = pin[1] + 0.5
except:
    row_pin = []
    col_pin = []

rows = len(data[0])
cols = int(len(data.T[0]) / 3)

mz = np.array([])
my = np.array([])
mx = np.array([])
for i in range(0, 3 * cols, 3):
    mz = np.concatenate([mz, data[i + 2]])

for i in range(0, 3 * cols, 3 * reduce_fac):
    my = np.concatenate([my, data[i + 1][::reduce_fac]])
    mx = np.concatenate([mx, data[i][::reduce_fac]])

mz = mz.reshape([cols, -1]).T
x = []
y = []
z = []

for j in range(0, cols, reduce_fac):
    for i in range(0, rows, reduce_fac):
        i_ = rows - i
        x.append(j + 0.5)
        y.append(i_ - 0.5)
        z.append(0)

x = np.array(x)
y = np.array(y)
z = np.array(z)

colors = ["gold", "white", "green"]
cmap1 = LinearSegmentedColormap.from_list("mcmp", colors)

r = rows / cols
FixPlot(8 / r, 8)
fig = plt.figure()
fig.set_size_inches(8 / r, 8 * 0.73 / 0.82)
ax = fig.add_axes([0.13, 0.15, 0.73, 0.82])


im = ax.imshow(mz, cmap="coolwarm", vmin=-1, vmax=1, extent=[0, cols, 0, rows], interpolation="none", aspect='auto')
divider = make_axes_locatable(ax)
cax1 = divider.append_axes("right", size="5%", pad=0.05)
bar = plt.colorbar(im, cax=cax1)
bar.set_label(fr"$m_z$", size=30)
bar.set_ticks([-1, 0, 1])
bar.ax.tick_params(labelsize=20)
print(bar.get_ticks())

nx = 4
hx = cols / nx
ny = 4
hy = rows / ny
ax.set_xticks([i * hx for i in range(nx + 1)])
ax.set_yticks([i * hy for i in range(ny + 1)])
ax.tick_params(axis='both', labelsize=20)
ax.set_xlabel("$x(a)$", size=30)
ax.set_ylabel("$y(a)$", size=30)

ax.quiver(x, y, mx, my, angles='xy', scale_units='xy', scale=np.sqrt(1) / reduce_fac, pivot="mid")

an = ax.scatter(col_ani, row_ani, color="green", s=20.0)
pi = ax.scatter(col_pin, row_pin, color="yellow", s=20.0)

plt.show()
fig.savefig("./imgs/out.png", dpi=500, facecolor="white", bbox_inches='tight')