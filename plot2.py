import numpy as np
from tvtk.api import tvtk
import matplotlib.pyplot as plt
from vtk import vtkStructuredGridReader
from vtk.util import numpy_support as VN
import mayavi.mlab as mlab
import moviepy.editor as mpy

reader = vtkStructuredGridReader()
reader.SetFileName("test.vtk")
reader.ReadAllVectorsOn()
reader.ReadAllScalarsOn()
reader.ReadAllFieldsOn()
reader.Update()

data = reader.GetOutput()

colors = VN.vtk_to_numpy(data.GetPointData().GetArray('rgbcolorstable'))
spins = VN.vtk_to_numpy(data.GetPointData().GetArray('spins'))
dim = data.GetDimensions()
nz = dim[2]
ny = dim[1]
nx = dim[0]
x, y, z = np.meshgrid(np.arange(nx), np.arange(ny), np.arange(nz))
spins = spins.reshape((nx, ny, nz, 3))
colors = colors.reshape((nx, ny, nz, 4))
colors = np.array(colors * 255, dtype=np.uint8)

mx = spins[:, :, :, 0]
my = spins[:, :, :, 1]
mz = spins[:, :, :, 2]

if 0:
    xyz = np.mgrid[0:nx, 0:ny, 0:nz]
    x = xyz[0]
    y = xyz[1]
    z = xyz[2]

    figure = mlab.figure(bgcolor=(1, 1, 1), size=(1000, 1000))
    field = mlab.pipeline.scalar_field(x, y, z, mz, vmin=-1.0, vmax=1.0)
    colour_data = colors.reshape((-1, 4))
    
    field.image_data.point_data.add_array(colour_data)
    field.image_data.point_data.get_array(1).name = 'phase'
    field.update()
    
    field2 = mlab.pipeline.set_active_attribute(field, point_scalars='scalar')
    contour = mlab.pipeline.contour(field2)
    contour.filter.contours = [0]
    contour2 = mlab.pipeline.set_active_attribute(contour, point_scalars='phase')
    
    mlab.pipeline.surface(contour2, vmin=-1 ,vmax=1)

if 0:
    print(dim)
    obj = mlab.quiver3d(x, y, z, mx, my, mz, scale_factor=3, mode="arrow")
    sc = tvtk.UnsignedCharArray()
    sc.from_array(colors[:, :, :, :3].reshape((-1, 3)))
    
    obj.mlab_source.dataset.point_data.scalars = sc
    obj.glyph.color_mode = "color_by_scalar"
    obj.mlab_source.dataset.modified()

mlab.show()
