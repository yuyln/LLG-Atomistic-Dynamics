import numpy as np
import mayavi.mlab as mlab
import utils
from tvtk.api import tvtk
from scipy import interpolate as interpolate

def square(x:np.ndarray, y: np.ndarray) -> np.ndarray:
    return np.cos(2.0 * np.pi * x * 6.0) + np.cos(2.0 * np.pi * y * 6)

x = np.linspace(0, 1, 1000)
y = np.linspace(0, 1, 1000)
x, y = np.mgrid[0:1:0.001, 0:1:0.001]
pot = square(x, y)
pot = pot * 0.05

figure = mlab.figure(bgcolor=(1, 1, 1), size=(1000, 1000))
mlab.surf(x, y, pot, colormap="coolwarm", vmin=pot.min(), vmax=pot.max())
for i in range(6):
    for j in range(6):


#field = mlab.pipeline.scalar_field(x, y, pot, vmin=-1.0, vmax=1.0)
#color_data = pot
#
#field.image_data.point_data.add_array(color_data.ravel())
#field.image_data.point_data.get_array(1).name = 'phase'
#field.update()
#
#field2 = mlab.pipeline.set_active_attribute(field, point_scalars='scalar')
#contour = mlab.pipeline.contour(field2)
#contour.filter.contours = [0.0]
#contour2 = mlab.pipeline.set_active_attribute(contour, point_scalars='phase')
#surf = mlab.pipeline.surface(contour2, colormap="hsv", vmin=-2, vmax=2.0)
mlab.show()
