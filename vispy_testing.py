from os import walk
import numpy as np

from vispy import app, scene, geometry, visuals
from scipy.spatial.transform import Rotation
import matplotlib as mpl
import utils
from scipy import interpolate as interpolate
from sklearn.preprocessing import PolynomialFeatures, SplineTransformer
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import Ridge, BayesianRidge

frames, gi, gp, raw = utils.ReadAnimationBinary("integrate_evolution.dat")

mx, my, mz = utils.GetFrameFromBinary(frames, gi, raw, frames - 1)
mx = np.transpose(mx.reshape((gi.depth, gi.rows, gi.cols)), (2, 1, 0))
my = np.transpose(my.reshape((gi.depth, gi.rows, gi.cols)), (2, 1, 0))
mz = np.transpose(mz.reshape((gi.depth, gi.rows, gi.cols)), (2, 1, 0))

x = np.arange(0, gi.cols)
y = np.arange(0, gi.rows)
z = np.arange(0, gi.depth)

x, y, z = np.meshgrid(x, y, z, indexing="ij")
xyz = np.transpose(np.concatenate([[x], [y], [z]]), (1, 2, 3, 0))

angle = np.atan2(my, mx)
angle = (angle - angle.min()) / (angle.max() - angle.min())
colors = mpl.colormaps["hsv"](angle)#.reshape((-1, 4))

canvas = scene.SceneCanvas(keys='interactive', bgcolor="white")
view = canvas.central_widget.add_view()

vertex, idx = geometry.isosurface.isosurface(mz, level=0)

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
#result0 = colors[((vx_idx + np.round(xd[:, 0])) % gi.cols).astype(int),
#                 ((vy_idx + np.round(yd[:, 0])) % gi.rows).astype(int), 
#                 ((vz_idx + np.round(zd[:, 0])) % gi.depth).astype(int)]

#interp0 = interpolate.LinearNDInterpolator(xyz.reshape((-1, 3)), colors)
#result0 = interp0((vertex[:, 0], vertex[:, 1], vertex[:, 2]))

shading_filter = visuals.filters.ShadingFilter(
        "smooth",
    # A shiny surface (small specular highlight).
    shininess=0,
    # A blue higlight, at half intensity.
    specular_coefficient=0,
    # Equivalent to `(0.7, 0.7, 0.7, 1.0)`.
    diffuse_coefficient=0,
    # Same as `(0.2, 0.3, 0.3, 1.0)`.
    ambient_coefficient=1,
    ambient_light = 1
)

mesh = scene.visuals.Mesh(vertex, idx, vertex_colors=result0)
#mesh.transform = scene.transforms.STTransform(translate=(-25, -25, -50))
mesh.attach(shading_filter)
view.add(mesh)

axis = scene.visuals.XYZAxis(parent=view.scene)

cam = scene.TurntableCamera(fov=45)
#cam.transform = scene.transforms.STTransform(translate=(0, 0, 0))
cam.set_range((-100, 100), (-100, 100), (-100, 100))
view.camera = cam

if __name__ == '__main__':
    canvas.show()
    app.run()
