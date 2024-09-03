import numpy as np
from vispy import app, scene, geometry, visuals, io
from vispy import util as vutils
import matplotlib as mpl
import utils

frames, gi, gp, raw = utils.ReadAnimationBinary("integrate_evolution.dat")

for frame in range(frames):
    mx, my, mz = utils.GetFrameFromBinary(frames, gi, raw, frame)
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
        
    shading_filter = visuals.filters.ShadingFilter(
        "smooth",
        shininess=0,
        specular_coefficient=0,
        diffuse_coefficient=0,
        ambient_coefficient=1,
        ambient_light = 1
    )
    
    mesh = scene.visuals.Mesh(vertex, idx, vertex_colors=result0)
    mesh.attach(shading_filter)
    view.add(mesh)
    
    cube = scene.visuals.Box(width=gi.cols, height=gi.depth, depth=gi.rows, color=(0, 0, 0, 0), edge_color=(0, 0, 0))
    cube.transform = scene.transforms.STTransform(translate=(gi.cols / 2, gi.rows / 2, gi.depth / 2))
    view.add(cube)
    
    cam = scene.TurntableCamera(fov=45, azimuth=45, elevation=45, distance=3 * gi.cols)
    cam.center = (gi.cols / 2, gi.rows / 2, gi.depth / 2)
    view.camera = cam
    canvas.draw()
    im = canvas.render()
    io.write_png(f"frame_{frame}.png", im)

