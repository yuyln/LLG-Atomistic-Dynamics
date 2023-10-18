import numpy as np
import utils
import matplotlib.pyplot as plt

rows = 1000
cols = 1000

t = np.linspace(0, 2.0 * np.pi, rows)
z = np.linspace(-1, 1, cols)
t, z = np.meshgrid(t, z)

mx = np.cos(t).reshape([-1])
my = np.sin(t).reshape([-1])
mz = z.reshape([-1])


rh, gh, bh = utils.GetHSL(mx, my, mz)
RGB = np.zeros([rows, cols, 3], dtype=float)
RGB[:, :, 0] = rh.reshape([rows, cols])
RGB[:, :, 1] = gh.reshape([rows, cols])
RGB[:, :, 2] = bh.reshape([rows, cols])
img = plt.imshow(RGB, origin="lower", extent=[0, 2 * np.pi, -1, 1])
plt.show()
