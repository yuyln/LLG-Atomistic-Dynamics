import numpy as np
from numpy import sin, cos, pi
from skimage import measure
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import utils
   
frames, gi, gp, raw = utils.ReadAnimationBinary("integrate_evolution.dat")

mx, my, mz = utils.GetFrameFromBinary(frames, gi, raw, 0)
mx = mx.reshape((gi.depth, gi.rows, gi.cols))
my = my.reshape((gi.depth, gi.rows, gi.cols))
mz = mz.reshape((gi.depth, gi.rows, gi.cols))
r, g, b = utils.GetHSL(mx, my, mz)
rgb = np.zeros((gi.depth, gi.rows, gi.cols, 3))
rgb[:, :, :, 0] = r
rgb[:, :, :, 1] = g
rgb[:, :, :, 2] = b

iso_val = 0.0
verts, faces, _, _ = measure.marching_cubes(mz, iso_val, spacing=(0.1, 0.1, 0.1))

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
a = ax.plot_trisurf(verts[:, 0], verts[:,1], faces, verts[:, 2])
plt.show()
