from Plooter import *
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.animation as anim
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from matplotlib import cm
from matplotlib.colors import LinearSegmentedColormap

reduce_fac = 1

data = pd.read_table("./output/end.out", header=None)
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

def GetM(data: pd.DataFrame) -> list[np.ndarray]:
    mx = np.array([])
    my = np.array([])
    mz = np.array([])
    cols = int(len(data.T[0]) / 3)
    for i in range(0, 3 * cols, 3):
        mz = np.concatenate([mz, data[i + 2]])
    for i in range(0, 3 * cols, 3 * reduce_fac):
        mx = np.concatenate([mx, data[i][::reduce_fac]])
        my = np.concatenate([my, data[i + 1][::reduce_fac]])
    return [mx, my, mz]

def GetBatch(full: pd.DataFrame, rows: int, index: int) -> pd.DataFrame:
    return full[index * rows: (index + 1) * rows].reset_index(drop=True)

def GetXY(data: pd.DataFrame) -> list[np.ndarray]:
    rows = len(data[0])
    cols = int(len(data.T[0]) / 3)
    x = []
    y = []
    for j in range(0, cols, reduce_fac):
        for i in range(0, rows, reduce_fac):
            x.append(j + 0.5)
            y.append(rows - i - 0.5)
    return [np.array(x), np.array(y)]


full = pd.read_table("./output/anim_grid.out", header=None)

b0 = GetBatch(full, rows, 0)
x, y = GetXY(b0)
mx, my, mz = GetM(b0)

r = rows / cols
FixPlot(8 / r, 8)
fig = plt.figure()
fig.set_size_inches(8 / r, 8 * 0.73 / 0.82)
ax = fig.add_axes([0.13, 0.15, 0.73, 0.82])

img = ax.imshow(mz.reshape([cols, -1]).T, cmap="coolwarm", vmin=-1, vmax=1, extent=[0, cols, 0, rows], interpolation='none', aspect='auto')
divider = make_axes_locatable(ax)
cax1 = divider.append_axes("right", size="5%", pad=0.05)
bar = plt.colorbar(img, cax=cax1)
bar.set_label(fr"$m_z$", size=30)
bar.set_ticks([-1, 0, 1])
bar.ax.tick_params(labelsize=20)
vecs = ax.quiver(x, y, mx, my, angles='xy', scale_units='xy', scale=np.sqrt(1) / reduce_fac, pivot="mid")
an = ax.scatter(col_ani, row_ani, color="green", s=50.0)
pi = ax.scatter(col_pin, row_pin, color="yellow", s=50.0)
nx = 4
hx = cols / nx
ny = 4
hy = rows / ny
ax.set_xticks([i * hx for i in range(nx + 1)])
ax.set_yticks([i * hy for i in range(ny + 1)])
ax.tick_params(axis='both', labelsize=20)
ax.set_xlabel("$x(a)$", size=30)
ax.set_ylabel("$y(a)$", size=30)
ax.set_xlim([0, cols])
ax.set_ylim([0, rows])

def animate(i):
    data = GetBatch(full, rows, i)
    mx, my, mz = GetM(data)
    x, y = GetXY(data)
    mz = mz.reshape([cols, -1]).T
    img.set_array(mz)
    vecs.set_UVC(mx, my)
frames = int(len(full[0]) / rows)
ani = anim.FuncAnimation(fig, animate, frames=frames)
ani.save("./videos/out.mp4", fps=60, dpi=100)
