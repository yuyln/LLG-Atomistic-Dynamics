import numpy as np
import moviepy.editor as mpy
import mayavi.mlab as mlab
import matplotlib as mpl
import utils
from tvtk.api import tvtk
from vispy import app, scene, geometry, visuals, io

frames, gi, gp, raw = utils.ReadAnimationBinaryF("integrate_evolution.dat")

mlab.options.offscreen = False
mlab.figure(fgcolor=(0, 0, 0), bgcolor=(1, 1, 1), size=(2000, 2000))
mlab.view(azimuth=45, elevation=45, distance=gi.cols / 2)
def draw_stuff(i):
    mlab.clf()
    mx, my, mz = utils.GetFrameFromBinary(frames, gi, raw, i)
    mx = np.transpose(mx.reshape((gi.depth, gi.rows, gi.cols)), (2, 1, 0))
    my = np.transpose(my.reshape((gi.depth, gi.rows, gi.cols)), (2, 1, 0))
    mz = np.transpose(mz.reshape((gi.depth, gi.rows, gi.cols)), (2, 1, 0))

    angle = np.arctan2(my, mx)
    angle = (angle - angle.min()) / (angle.max() - angle.min())
    colors = mpl.colormaps["hsv"](angle)
    
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
    
    scalars = np.arange(result0.reshape((-1, 4)).shape[0])
    mesh = mlab.triangular_mesh(vertex[:, 0], vertex[:, 1], vertex[:, 2], idx, scalars=scalars) 
    mesh.module_manager.scalar_lut_manager.lut.table = (result0.reshape((-1, 4)) * 255).astype(np.uint8)

    mlab.plot3d([0, gi.cols], [0, 0], [0, 0], color=(0, 0, 0), tube_radius=1.)
    mlab.plot3d([0, gi.cols], [gi.rows, gi.rows], [0, 0], color=(0, 0, 0), tube_radius=1.)
    mlab.plot3d([0, 0], [0, gi.rows], [0, 0], color=(0, 0, 0), tube_radius=1.)
    mlab.plot3d([gi.cols, gi.cols], [0, gi.rows], [0, 0], color=(0, 0, 0), tube_radius=1.)

    mlab.plot3d([0, gi.cols], [0, 0], [gi.depth, gi.depth], color=(0, 0, 0), tube_radius=1.)
    mlab.plot3d([0, gi.cols], [gi.rows, gi.rows], [gi.depth, gi.depth], color=(0, 0, 0), tube_radius=1.)
    mlab.plot3d([0, 0], [0, gi.rows], [gi.depth, gi.depth], color=(0, 0, 0), tube_radius=1.)
    mlab.plot3d([gi.cols, gi.cols], [0, gi.rows], [gi.depth, gi.depth], color=(0, 0, 0), tube_radius=1.)


    mlab.plot3d([0, 0], [0, 0], [0, gi.depth], color=(0, 0, 0), tube_radius=1.)
    mlab.plot3d([gi.cols, gi.cols], [0, 0], [0, gi.depth], color=(0, 0, 0), tube_radius=1.)
    mlab.plot3d([0, 0], [gi.rows, gi.rows], [0, gi.depth], color=(0, 0, 0), tube_radius=1.)
    mlab.plot3d([gi.cols, gi.cols], [gi.rows, gi.rows], [0, gi.depth], color=(0, 0, 0), tube_radius=1.)

if mlab.options.offscreen:
    duration = 2
    def make_frame(t):
        i = int(t / duration * frames)
        print(i)
        draw_stuff(i)
        return mlab.screenshot()
    
    animation = mpy.VideoClip(make_frame, duration=duration)
    animation.write_videofile("test.mp4", fps=60)
    exit(1)


draw_stuff(frames - 1)
mlab.show()
