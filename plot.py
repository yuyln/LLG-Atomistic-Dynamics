import utils
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

utils.FixPlot(8, 8)
cmd = utils.CMDArgs("dummy", "dummy")
frames, gi, gp, raw = utils.ReadAnimationBinary("integrate_evolution.dat")

fig, ax = plt.subplots()

min_x, max_x = 0, gi.cols * gi.lattice
min_y, max_y = 0, gi.rows * gi.lattice
min_z, max_z = 0, gi.depth * gi.lattice
sx, sy, sz = max_x - min_x, max_y - min_y, max_z - min_z

mx, my, mz = utils.GetFrameFromBinary(frames, gi, raw, frames - 1)
mx = mx.reshape((gi.depth, gi.rows, gi.cols))
my = my.reshape((gi.depth, gi.rows, gi.cols))
mz = mz.reshape((gi.depth, gi.rows, gi.cols))

for i in range(gi.rows):
    mx_ = mx[:, i, :]
    my_ = my[:, i, :]
    mz_ = mz[:, i, :]

    r, g, b = utils.GetHSL(mx_, my_, mz_ * cmd.INVERT)
    rgb = np.zeros((*mx_.shape, 3))
    rgb[:, :, 0] = r.reshape((mx_.shape))
    rgb[:, :, 1] = g.reshape((mx_.shape))
    rgb[:, :, 2] = b.reshape((mx_.shape))
    plt.imshow(my_, origin="lower", cmap=utils.cmap, vmax=1, vmin=-1, interpolation="bicubic")
    plt.show()

