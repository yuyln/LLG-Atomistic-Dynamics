import utils
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import mayavi.mlab as mlab
import moviepy.editor as mpy
from tvtk.api import tvtk

frames, gi, gp, raw = utils.ReadAnimationBinary("integrate_evolution.dat")

z, y, x = np.meshgrid(np.arange(gi.cols), np.arange(gi.depth), np.arange(gi.rows))
mx, my, mz = utils.GetFrameFromBinary(frames, gi, raw, frames - 1)
mx = np.transpose(mx.reshape((gi.depth, gi.rows, gi.cols)), (2, 1, 0))
my = np.transpose(my.reshape((gi.depth, gi.rows, gi.cols)), (2, 1, 0))
mz = np.transpose(mz.reshape((gi.depth, gi.rows, gi.cols)), (2, 1, 0))

for i in range(gi.rows):
    plt.imshow(my[:, i, :].T, cmap="bwr", vmin=-1, vmax=1)
    plt.show()
