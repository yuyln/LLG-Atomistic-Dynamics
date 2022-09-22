from Plooter import *
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
import array
import pandas as pd

REDUCE_FACTOR = 1

file = open("./output/end.bin", "rb")
raw_data = file.read()
file.close()

nrow_ncol = array.array("i")
nrow_ncol.frombytes(raw_data[:8])
nrows = nrow_ncol[0]
ncols = nrow_ncol[1]

raw_vec = array.array("d")
raw_vec.frombytes(raw_data[8:])
M = np.array(raw_vec)
mx_, my_, mz = M[0::3], M[1::3], M[2::3]
xs, ys = zip(*[[j, i] for i in range(nrows) for j in range(ncols)])
xs = np.array(xs)
ys = np.array(ys)
xu = np.unique(xs)[::REDUCE_FACTOR]
yu = np.unique(ys)[::REDUCE_FACTOR]

x_in = np.in1d(xs, xu)

y_in = ys[x_in]
y_in = np.in1d(y_in, yu)

mx = mx_[x_in]
mx = mx[y_in]

my = my_[x_in]
my = my[y_in]

x = xs[x_in]
x = x[y_in]

y = ys[x_in]
y = y[y_in]

facx = xu[1] - xu[0]
facy = yu[1] - yu[0]

r = nrows / ncols
FixPlot(8 / r, 8)
fig = plt.figure()
fig.set_size_inches(8 / r, 8 * 0.73 / 0.82)
ax = fig.add_axes([0.13, 0.15, 0.73, 0.82])

img = ax.imshow(mz.reshape([nrows, ncols]), cmap="coolwarm", vmin=-1.0, vmax=1.0, origin="lower")
divider = make_axes_locatable(ax)
cax1 = divider.append_axes("right", size="5%", pad=0.05)
bar = plt.colorbar(img, cax=cax1)
bar.set_label(fr"$m_z$", size=30)
bar.set_ticks([-1, 0, 1])
bar.ax.tick_params(labelsize=20)

vecs = ax.quiver(x, y, mx * facx, my * facy, angles='xy', scale_units='xy', pivot="mid", scale=1)

nx = 4
hx = ncols / nx
ny = 4
hy = nrows / ny
ax.set_xticks([i * hx for i in range(nx + 1)])
ax.set_yticks([i * hy for i in range(ny + 1)])
ax.tick_params(axis='both', labelsize=20)
ax.set_xlabel("$x(a)$", size=30)
ax.set_ylabel("$y(a)$", size=30)

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

an = ax.scatter(col_ani, row_ani, color="green", s=20.0)
pi = ax.scatter(col_pin, row_pin, color="yellow", s=20.0)
ax.set_xlim([-0.5, ncols - 0.5])
ax.set_ylim([-0.5, nrows - 0.5])
plt.show()
fig.savefig("./imgs/out_bin.png", dpi=500, facecolor="white", bbox_inches='tight')