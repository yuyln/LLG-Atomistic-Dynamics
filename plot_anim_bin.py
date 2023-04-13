from Plooter import *
import matplotlib.pyplot as plt
import matplotlib.animation as anim
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import array
import pandas as pd
REDUCE_FACTOR = 1
PLOT_ARROWS = False 
PLOT_ANI = False
PLOT_PIN = False
INTERPOLATION = "nearest"

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

# position = pd.read_table("./output/anim_pos_xy.out", header=None, delimiter="\t")
# position[1] /= 0.5e-9
# position[2] /= 0.5e-9
# position[1] += 0.5
# position[2] += 0.5

file = open("./output/integration_fly.bin", "rb")
raw_data = file.read()
file.close()
nrow_ncol_steps = array.array("i")
nrow_ncol_steps.frombytes(raw_data[:12])

nrows = nrow_ncol_steps[0]
ncols = nrow_ncol_steps[1]
total = nrows * ncols
frames = nrow_ncol_steps[2]

xs, ys = zip(*[[j, i] for i in range(nrows) for j in range(ncols)])
xs = np.array(xs)
ys = np.array(ys)
xu = np.unique(xs)[::REDUCE_FACTOR]
yu = np.unique(ys)[::REDUCE_FACTOR]

x_in = np.in1d(xs, xu)

y_in = ys[x_in]
y_in = np.in1d(y_in, yu)

x = xs[x_in]
x = x[y_in]

y = ys[x_in]
y = y[y_in]

facx = xu[1] - xu[0]
facy = yu[1] - yu[0]

vecsz = 3 * 8
def GetBatch(index:int):
    if index >= frames: index = frames - 1
    elif index < 0: index = 0

    raw_vecs = array.array("d")
    raw_vecs.frombytes(raw_data[12 + index * total * vecsz: 12 + (index + 1) * total * vecsz])
    M = np.array(raw_vecs)
    mx_, my_, mz_ = M[0::3], M[1::3], M[2::3]
    mx__ = mx_
    my__ = my_

    mx = mx__[x_in]
    mx = mx[y_in]

    my = my__[x_in]
    my = my[y_in]

    mz = mz_
    return mx * facx, my * facy, mz

mx, my, mz = GetBatch(0)

r = nrows / ncols
FixPlot(8 / r, 8)
fig = plt.figure()
fig.set_size_inches(8 / r, 8 * 0.73 / 0.82)
ax = fig.add_axes([0.13, 0.15, 0.73, 0.82])

colors = ["#037fff", "white", "#f40501"]
cmap1 = LinearSegmentedColormap.from_list("mcmp", colors)

img = ax.imshow(mz.reshape([nrows, ncols]), cmap=cmap1, vmin=-1.0, vmax=1.0, origin="lower", interpolation=INTERPOLATION)
divider = make_axes_locatable(ax)
cax1 = divider.append_axes("right", size="5%", pad=0.05)
bar = plt.colorbar(img, cax=cax1)
bar.set_label(fr"$m_z$", size=28)
bar.set_ticks([-1, 0, 1])
bar.ax.tick_params(labelsize=22)

nx = 4
hx = ncols / nx
ny = 4
hy = nrows / ny
ax.set_xticks([i * hx for i in range(nx + 1)])
ax.set_yticks([i * hy for i in range(ny + 1)])
ax.tick_params(axis='both', labelsize=22)
ax.set_xlabel("$x(a)$", size=28)
ax.set_ylabel("$y(a)$", size=28)

if PLOT_ARROWS:
    global vecs
    vecs = ax.quiver(x, y, mx, my, angles='xy', scale_units='xy', pivot="mid", scale=1, width=0.002)


if PLOT_ANI: an = ax.scatter(col_ani, row_ani, color="green", s=10.0)
if PLOT_PIN: pi = ax.scatter(col_pin, row_pin, color="yellow", s=10.0)
ax.set_xlim([-0.5, ncols - 0.5])
ax.set_ylim([-0.5, nrows - 0.5])

# center = ax.scatter(position[1][0], position[2][0], s=50.0, color="black")
# frames_pos = len(position[0])
# ratio = int(float(frames_pos) / float(frames))

def animate(i):
    mx, my, mz = GetBatch(i)
    mz = mz.reshape([nrows, ncols])
    img.set_array(mz)
    if PLOT_ARROWS: vecs.set_UVC(mx, my)
    # center.set_offsets([[position[1][ratio * i], position[2][ratio * i]]])


print(f"Total frames: {frames}")
ani = anim.FuncAnimation(fig, animate, frames=frames)
ani.save("./videos/out_bin.mp4", fps=60, dpi=250)
