import utils
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import mayavi.mlab as mlab
import moviepy.editor as mpy
from tvtk.api import tvtk

frames, gi, gp, raw = utils.ReadAnimationBinary("integrate_evolution.dat")

z, y, x = np.meshgrid(np.arange(gi.cols), np.arange(gi.depth), np.arange(gi.rows))

duration = 10.0
mlab.options.offscreen = True
def make_frame(t):
    i = int(t / duration * frames)
    mx, my, mz = utils.GetFrameFromBinary(frames, gi, raw, i)
    r, g, b = utils.GetHSL(mx, my, mz)
    rgb = np.zeros((gi.depth, gi.rows, gi.cols, 3))
    rgb[:, :, :, 0] = r.reshape(rgb.shape[:3])
    rgb[:, :, :, 1] = g.reshape(rgb.shape[:3])
    rgb[:, :, :, 2] = b.reshape(rgb.shape[:3])
    rgb = np.array(rgb * 255, dtype=int)
    mx = mx.reshape(rgb.shape[:3])
    my = my.reshape(rgb.shape[:3])
    mz = mz.reshape(rgb.shape[:3])
    zero = np.where(mz > 0.0)
    mx[zero] = 0
    my[zero] = 0
    mz[zero] = 0
    
    fig = mlab.figure(size=(1000, 1000))
    mlab.clf()

    obj = mlab.quiver3d(x, y, z, mx, my, mz, scale_factor=3, mode="arrow")
    sc = tvtk.UnsignedCharArray()
    sc.from_array(rgb.reshape((-1, 3)))
    
    obj.mlab_source.dataset.point_data.scalars = sc
    obj.glyph.color_mode = "color_by_scalar"
    obj.mlab_source.dataset.modified()

    mlab.view(azimuth=20, elevation=20, distance=300) # camera angle

    imgmap = mlab.screenshot(figure=fig, mode='rgb', antialiased=True)
    print(imgmap.shape)
    mlab.close(fig)
    return imgmap

animation = mpy.VideoClip(make_frame, duration=duration)
print(animation.w, animation.h)
animation.write_videofile("test.mp4", fps=60)
