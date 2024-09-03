import numpy as np
import mayavi.mlab as mlab
import matplotlib as mpl
import utils
from tvtk.api import tvtk
from vispy import app, scene, geometry, visuals, io

frames, gi, gp, raw = utils.ReadAnimationBinary("integrate_evolution.dat.bak")

mx, my, mz = utils.GetFrameFromBinary(frames, gi, raw, frames - 1)
mx = np.transpose(mx.reshape((gi.depth, gi.rows, gi.cols)), (2, 1, 0))
my = np.transpose(my.reshape((gi.depth, gi.rows, gi.cols)), (2, 1, 0))
mz = np.transpose(mz.reshape((gi.depth, gi.rows, gi.cols)), (2, 1, 0))

mlab.figure(1, fgcolor=(1, 1, 1), bgcolor=(0, 0, 0))
x, y, z = np.meshgrid(np.arange(gi.cols), np.arange(gi.rows), np.arange(gi.depth), indexing="ij")

angle = np.arctan2(my, mx)
angle = (angle - angle.min()) / (angle.max() - angle.min())
colors = mpl.colormaps["hsv"](angle)

vertex, idx = geometry.isosurface.isosurface(mz, level=-0.5)

vx_idx = vertex[:, 0].astype(int)
vy_idx = vertex[:, 1].astype(int)
vz_idx = vertex[:, 2].astype(int)
xd = (vertex[:, 0] - vx_idx)
yd = (vertex[:, 1] - vy_idx)
zd = (vertex[:, 2] - vz_idx)

xd = np.column_stack((xd, xd, xd, xd))
yd = np.column_stack((yd, yd, yd, yd))
zd = np.column_stack((zd, zd, zd, zd))
c00 = colors[vx_idx, vy_idx, vz_idx] * (1 - xd) + colors[(vx_idx + 1) % gi.cols, vy_idx, vz_idx] * xd
c10 = colors[vx_idx, (vy_idx + 1) % gi.rows, vz_idx] * (1 - xd) + colors[(vx_idx + 1) % gi.cols, (vy_idx + 1) % gi.rows, vz_idx] * xd
c01 = colors[vx_idx, vy_idx, (vz_idx + 1) % gi.depth] * (1 - xd) + colors[(vx_idx + 1) % gi.cols, vy_idx, (vz_idx + 1) % gi.depth] * xd
c11 = colors[vx_idx, (vy_idx + 1) % gi.rows, (vz_idx + 1) % gi.depth] * (1 - xd) + colors[(vx_idx + 1) % gi.cols, (vy_idx + 1) % gi.rows, (vz_idx + 1) % gi.depth] * xd

c0 = c00 * (1 - yd) + c10 * yd
c1 = c01 * (1 - yd) + c11 * yd
result0 = c0 * (1 - zd) + c1 * zd

scalars = np.arange(result0.reshape((-1, 4)).shape[0])
mesh = mlab.triangular_mesh(vertex[:, 0], vertex[:, 1], vertex[:, 2], idx, scalars=scalars) 
mesh.module_manager.scalar_lut_manager.lut.table = (result0.reshape((-1, 4)) * 255).astype(np.uint8)

mlab.show()
