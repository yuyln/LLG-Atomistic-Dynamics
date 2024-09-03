import utils
#import numpy as np
#import pyvista as pv
#
#data = pv.read("./test.vtk")
#colors = pv.read_texture("./colormap.png")
#colors.interpolation = False
#angles = data.get_array("angle")
#contours = data.contour([0.0], "mz_table")
#contours = contours.clean(tolerance=1e-12)
#
#p = pv.Plotter()
##p.add_mesh(contours.arrows, scalars="angle", colormap="hsv", interpolate_before_map=True, lighting=True, clim=[-np.pi, np.pi])
#
##actor = p.add_mesh(contours, scalars="angle", colormap="hsv", interpolate_before_map=True, lighting=True, clim=[-np.pi, np.pi])
#p.add_mesh(contours, opacity=1, style='surface',
#           scalars='angle',
#           show_scalar_bar=False,
#           cmap='hsv',  # use cmocean phase colormap
#           clim=[-np.pi, np.pi],
#           interpolate_before_map=False,
#           smooth_shading=True, lighting=True, specular=0.6, specular_power=10, ambient=0.
#          )
##actor = p.add_mesh(contours, texture=colors, interpolate_before_map=False, lighting=False)
##prop = actor.GetProperty()
##prop.interpolation = "Gouraud"
#p.show()
#exit(1)


import pyvista as pv
import numpy as np

def create_hopfion(nx: int, ny: int, nz: int, c1: float, R: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x = np.arange(nx, dtype=float)
    y = np.arange(ny, dtype=float)
    z = np.arange(nz, dtype=float)
    x -= nx / 2.0
    y -= ny / 2.0
    z -= nz / 2.0
    z, y, x = np.meshgrid(z, y, x, indexing="ij")
    r = np.sqrt(x ** 2 + y ** 2 + z ** 2) + 1e-15
    f = np.exp(-r ** 2.0 / (2.0 * 100) ** 2.0) * np.pi
    f = np.asin(2.0 * r / (r ** 2 + 1))
    my = x / r * np.sin(2 * f) + y * z / r ** 2 * np.sin(f) ** 2
    mx = y / r * np.sin(2.0 * f) - x * z / r ** 2 * np.sin(f) ** 2
    mz = np.cos(2.0 * f) + 2.0 * z ** 2.0 / r ** 2.0 * np.sin(f) ** 2
    
    return np.arange(nx), np.arange(ny), np.arange(nz), mx, my, mz

def create_hopfion2(nx: int, ny: int, nz: int, R: float, h: float, wr: float, wh: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x = np.arange(nx, dtype=float)
    y = np.arange(ny, dtype=float)
    z = np.arange(nz, dtype=float)
    x -= nx / 2.0
    y -= ny / 2.0
    z -= nz / 2.0
    z, y, x = np.meshgrid(z, y, x, indexing="ij")
    phi = np.atan2(y, x)
    r = np.sqrt(x ** 2 + y ** 2) + 1e-15
    rl = (np.exp(R / wr) - 1) / (np.exp(r / wr) -1)
    zl = z / np.abs(z + 1.0e-15) * (np.exp(np.abs(z) / wh) - 1) / (np.exp(h / wh) - 1)
    mx = 4.0 * rl * (2.0 * zl * np.sin(phi) + (rl ** 2 + zl ** 2 + 1) * np.cos(phi)) / (1 + rl ** 2 + zl ** 2) ** 2
    my = 4.0 * rl * (-2.0 * zl * np.cos(phi) + (rl ** 2 + zl ** 2 + 1) * np.sin(phi)) / (1 + rl ** 2 + zl ** 2) ** 2
    mz = 1 - 8 * rl ** 2 / (1 + rl ** 2 + zl ** 2) ** 2
    
    return np.arange(nx), np.arange(ny), np.arange(nz), mx, my, mz

def create_skyrmion(nx: int, ny: int, nz: int, dw: float, R: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x = np.arange(nx, dtype=float)
    y = np.arange(ny, dtype=float)
    z = np.arange(nz, dtype=float)
    x -= nx / 2.0
    y -= ny / 2.0
    z -= nz / 2.0
    z, y, x = np.meshgrid(z, y, x, indexing="ij")
    r = np.sqrt(x * x + y * y) #polar
    phi = np.arctan2(y, x) + np.pi
    theta = 2.0 * np.arctan(np.sinh(R / dw) / np.sinh(r / dw))
    mx = np.cos(phi) * np.sin(theta)
    my = np.sin(phi) * np.sin(theta)
    mz = np.cos(theta)
    return np.arange(nx), np.arange(ny), np.arange(nz), mx, my, mz

frames, gi, gp, raw = utils.ReadAnimationBinary("integrate_evolution.dat.bak")
#mx, my, mz = utils.GetFrameFromBinary(frames, gi, raw, frames - 1)
#mx = mx.reshape((gi.depth, gi.rows, gi.cols))
#my = my.reshape((gi.depth, gi.rows, gi.cols))
#mz = mz.reshape((gi.depth, gi.rows, gi.cols))

x, y, z, mx, my, mz = create_hopfion(gi.cols, gi.rows, gi.depth, 1, 1)
dataset = pv.RectilinearGrid(x, y, z)

angle = np.arctan2(my, mx)
v = np.zeros((*mx.shape, 3))
v[:, :, :, 0] = mx
v[:, :, :, 1] = my
v[:, :, :, 2] = mz

angle = (angle - angle.min()) / (angle.max() - angle.min())

dataset.point_data["angle"] = angle.ravel()

dataset.point_data["mz_table"] = mz.ravel()
dataset.point_data["vectors"] = v.reshape((-1, 3))
dataset.set_active_vectors("vectors")

surface = dataset.contour([0.], scalars='mz_table')

surface.point_data['sin'] = np.sin(surface.point_data['angle'])
surface.point_data['cos'] = np.cos(surface.point_data['angle'])

# Subdivision automatically does linear interpolation on the scalar values
subdivided = surface.subdivide_adaptive(max_n_passes=200)  # Whatever area or edge length is appropriate
subdivided.point_data['angle'] = np.arctan2(subdivided.point_data['sin'], subdivided.point_data['cos'])

p = pv.Plotter()
p.add_mesh(surface,
           scalars='angle',
           show_scalar_bar=False,
           cmap='hsv',
           clim=[0, 1],
           interpolate_before_map=False)

#p.add_mesh(ribbon,
#           show_scalar_bar=False,
#           interpolate_before_map=False,
#           color="black",
#           line_width=10)

#p.add_mesh(surface.arrows,
#           scalars='angle',
#           show_scalar_bar=False,
#           cmap='hsv',
#           clim=[-np.pi, np.pi],
#           interpolate_before_map=True)

p.show()
