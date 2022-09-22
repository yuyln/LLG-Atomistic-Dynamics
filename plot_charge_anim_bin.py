from Plooter import *
import matplotlib.pyplot as plt
import matplotlib.animation as anim
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
import array
import pandas as pd
REDUCE_FACTOR = 1

try:
    ani = pd.read_table("./input/anisotropy.in", header=None, delimiter=" ", skiprows=2)
    row_ani = ani[0]
    col_ani = ani[1]
except:
    row_ani = []
    col_ani = []

try:
    pin = pd.read_table("./input/pinning.in", header=None, delimiter=" ", skiprows=2)
    row_pin = pin[0]
    col_pin = pin[1]
except:
    row_pin = []
    col_pin = []

file = open("./output/grid_charge_anim.bin", "rb")
raw_data = file.read()
file.close()
nrow_ncol_steps = array.array("i")
nrow_ncol_steps.frombytes(raw_data[:12])

nrows = nrow_ncol_steps[0]
ncols = nrow_ncol_steps[1]
total = nrows * ncols
frames = nrow_ncol_steps[2]

raw_vecs = array.array("d")
raw_vecs.frombytes(raw_data[12:])
M = np.array(raw_vecs)


def GetBatch(index:int):
    if index >= frames: index = frames - 1
    elif index < 0: index = 0
    return M[index * total:(index + 1) * total]

mz = GetBatch(0)

r = nrows / ncols
FixPlot(8 / r, 8)
fig = plt.figure()
fig.set_size_inches(8 / r, 8 * 0.73 / 0.82)
ax = fig.add_axes([0.13, 0.15, 0.73, 0.82])

img = ax.imshow(mz.reshape([nrows, ncols]), cmap="coolwarm", origin="lower")
divider = make_axes_locatable(ax)
cax1 = divider.append_axes("right", size="5%", pad=0.05)
bar = plt.colorbar(img, cax=cax1)
bar.set_label(fr"$Q$", size=30)
bar.ax.tick_params(labelsize=20)

nx = 4
hx = ncols / nx
ny = 4
hy = nrows / ny
ax.set_xticks([i * hx for i in range(nx + 1)])
ax.set_yticks([i * hy for i in range(ny + 1)])
ax.tick_params(axis='both', labelsize=20)
ax.set_xlabel("$x(a)$", size=30)
ax.set_ylabel("$y(a)$", size=30)


an = ax.scatter(col_ani, row_ani, color="green", s=20.0)
pi = ax.scatter(col_pin, row_pin, color="yellow", s=20.0)
ax.set_xlim([-0.5, ncols - 0.5])
ax.set_ylim([-0.5, nrows - 0.5])

def animate(i):
    mz = GetBatch(i)
    mz = mz.reshape([nrows, ncols])
    img.set_array(mz)

ani = anim.FuncAnimation(fig, animate, frames=frames)
ani.save("./videos/out_bin_charge.mp4", fps=60, dpi=100)
