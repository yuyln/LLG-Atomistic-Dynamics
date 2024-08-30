import numpy as np
import mayavi.mlab as mlab
import utils
from tvtk.api import tvtk
from scipy import interpolate as interpolate

frames, gi, gp, raw = utils.ReadAnimationBinary("integrate_evolution.dat")

mx, my, mz = utils.GetFrameFromBinary(frames, gi, raw, frames - 1)
mx = mx.reshape((gi.depth, gi.rows, gi.cols))
my = my.reshape((gi.depth, gi.rows, gi.cols))
mz = mz.reshape((gi.depth, gi.rows, gi.cols))

r, g, b = utils.GetHSL(mx, my, mz)
rgb = np.zeros((gi.depth, gi.rows, gi.cols, 4))
rgb[:, :, :, 0] = r.reshape(mx.shape)
rgb[:, :, :, 1] = g.reshape(mx.shape)
rgb[:, :, :, 2] = b.reshape(mx.shape)
rgb[:, :, :, 3] = 1
rgb = np.array(rgb * 255, dtype=np.uint8)
colors = rgb

z, y, x = np.meshgrid(np.arange(gi.depth), np.arange(gi.rows), np.arange(gi.cols), indexing="ij")
#print(gi)
#
figure = mlab.figure(bgcolor=(1, 1, 1), size=(1000, 1000))
field = mlab.pipeline.scalar_field(mz, vmin=-1.0, vmax=1.0)
color_data = np.arctan2(my, mx).reshape((-1)) + np.pi

interp0 = interpolate.NearestNDInterpolator((x.ravel(),y.ravel(),z.ravel()), color_data.ravel())
result0 = interp0((x, y, z))

field.image_data.point_data.add_array(result0.ravel())
field.image_data.point_data.get_array(1).name = 'phase'
field.update()

field2 = mlab.pipeline.set_active_attribute(field, point_scalars='scalar')
contour = mlab.pipeline.contour(field2)
contour.filter.contours = [0.0]
contour2 = mlab.pipeline.set_active_attribute(contour, point_scalars='phase')
surf = mlab.pipeline.surface(contour2, colormap="hsv", vmin=0, vmax=2.0 * np.pi)


#where = np.where(mz > 0.5)
#mx[where] *= 0
#my[where] *= 0
#mz[where] *= 0
#
#z, y, x = np.meshgrid(np.arange(gi.depth), np.arange(gi.rows), np.arange(gi.cols), indexing="ij")
#print(z.shape)
#print(mx.shape)
#
#obj = mlab.quiver3d(x, y, z, mx, my, mz, scale_factor=3, mode="arrow")
#sc = tvtk.UnsignedCharArray()
#sc.from_array(colors[:, :, :, :3].reshape((-1, 3)))
#
#obj.mlab_source.dataset.point_data.scalars = sc
#obj.glyph.color_mode = "color_by_scalar"
#obj.mlab_source.dataset.modified()

mlab.show()
