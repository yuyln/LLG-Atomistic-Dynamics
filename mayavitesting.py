## Create the data ############################################################
#import numpy as np
#
#x, y, z = np.ogrid[- .5:.5:20j, - .5:.5:20j, - .5:.5:20j]
#r = np.sqrt(x ** 2 + y ** 2 + z ** 2)
#
#
## Generalized Laguerre polynomial (3, 2)
#L = - r ** 3 / 6 + 5. / 2 * r ** 2 - 10 * r + 6
#
## Spherical harmonic (3, 2)
#Y = (x + y * 1j) ** 2 * z / r ** 3
#
#Phi = L * Y * np.exp(- r) * r ** 2
#
#Phi = r
#
## Plot it ####################################################################
#from mayavi import mlab
#mlab.figure(1, fgcolor=(1, 1, 1), bgcolor=(0, 0, 0))
## We create a scalar field with the module of Phi as the scalar
#src = mlab.pipeline.scalar_field(np.abs(Phi))
#
## And we add the phase of Phi as an additional array
## This is a tricky part: the layout of the new array needs to be the same
## as the existing dataset, and no checks are performed. The shape needs
## to be the same, and so should the data. Failure to do so can result in
## segfaults.
#t = np.arctan2(y, x, dtype=float)
#t = np.stack([t for _ in range(20)], axis=-1)
#src.image_data.point_data.add_array(t.T.ravel())
## We need to give a name to our new dataset.
#src.image_data.point_data.get_array(1).name = 'angle'
## Make sure that the dataset is up to date with the different arrays:
#src.update()
#
## We select the 'scalar' attribute, ie the norm of Phi
#src2 = mlab.pipeline.set_active_attribute(src,
#                                    point_scalars='scalar')
#
## Cut isosurfaces of the norm
#contour = mlab.pipeline.contour(src2)
#
## Now we select the 'angle' attribute, ie the phase of Phi
#contour2 = mlab.pipeline.set_active_attribute(contour,
#                                    point_scalars='angle')
#
## And we display the surface. The colormap is the current attribute: the phase.
#mlab.pipeline.surface(contour2, colormap='hsv')
#
#mlab.colorbar(title='Phase', orientation='vertical', nb_labels=3)
#
#mlab.show()
#exit(1)

import numpy as np
import mayavi.mlab as mlab
import mayavi.tools
import utils
from tvtk.api import tvtk
from scipy import interpolate as interpolate
import colorsys
import scipy.interpolate

frames, gi, gp, raw = utils.ReadAnimationBinary("integrate_evolution.dat.bak")

rgb_to_hex = lambda x: int(f"{x[0]:02X}{x[1]:02X}{x[2]:02X}ff", 16)

mx, my, mz = utils.GetFrameFromBinary(frames, gi, raw, frames - 1)
mx = np.transpose(mx.reshape((gi.depth, gi.rows, gi.cols)), (2, 1, 0))
my = np.transpose(my.reshape((gi.depth, gi.rows, gi.cols)), (2, 1, 0))
mz = np.transpose(mz.reshape((gi.depth, gi.rows, gi.cols)), (2, 1, 0))

mlab.figure(1, fgcolor=(1, 1, 1), bgcolor=(0, 0, 0))
x, y, z = np.meshgrid(np.linspace(-1, 1, gi.cols), np.linspace(-1, 1, gi.rows), np.linspace(-1, 1, gi.depth))
src = mlab.pipeline.scalar_field(np.sqrt(x * x + y * y + z * z))

t = np.arctan2(y, x, dtype=float) * 2
w = np.where(np.abs(t) > np.pi)
t[w] = t[w] - np.floor((t[w] + np.pi) / (2.0 * np.pi)) * np.pi
src.image_data.point_data.add_array(t.T.ravel())
# We need to give a name to our new dataset.
src.image_data.point_data.get_array(1).name = 'angle'
# Make sure that the dataset is up to date with the different arrays:
src.update()

# We select the 'scalar' attribute, ie the norm of Phi
src2 = mlab.pipeline.set_active_attribute(src,
                                    point_scalars='scalar')

# Cut isosurfaces of the norm
contour = mlab.pipeline.contour(src2)

# Now we select the 'angle' attribute, ie the phase of Phi
contour2 = mlab.pipeline.set_active_attribute(contour,
                                    point_scalars='angle')

# And we display the surface. The colormap is the current attribute: the phase.
mlab.pipeline.surface(contour2, colormap='hsv')

mlab.colorbar(title='Phase', orientation='vertical', nb_labels=3)

mlab.show()
