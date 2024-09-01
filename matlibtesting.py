import numpy as np
from numpy import cos, pi
from skimage.measure import marching_cubes
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import utils

frames, gi, gp, raw = utils.ReadAnimationBinary("integrate_evolution.dat.bak")

mx, my, mz = utils.GetFrameFromBinary(frames, gi, raw, frames - 1)
mx = mx.reshape((gi.depth, gi.rows, gi.cols))
my = my.reshape((gi.depth, gi.rows, gi.cols))
mz = mz.reshape((gi.depth, gi.rows, gi.cols))

z, y, x = np.meshgrid(np.arange(gi.depth) * 1.0, np.arange(gi.rows) * 1.0, np.arange(gi.cols) * 1.0, indexing="ij")

iso_val=0.0
verts, faces, _, _ = marching_cubes(mz, iso_val, spacing=(1, 1, 1))

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_trisurf(verts[:, 0], verts[:,1], faces, verts[:, 2], cmap='Spectral',
                lw=1)
ax.set_xlim((0, gi.depth))
ax.set_ylim((0, gi.rows))
ax.set_zlim((0, gi.cols))
plt.show()
