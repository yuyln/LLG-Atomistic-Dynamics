import utils
import matplotlib.pyplot as plt
import numpy as np

utils.FixPlot_(8, 8)
fig, ax = plt.subplots()

N = 10000
rows = N
cols = N

x = np.linspace(-2, 2, N)
y = np.linspace(-2, 2, N)
x, y = np.meshgrid(x, y)
theta = np.arctan2(y, x) / np.pi
theta = (theta + 1) / 2.0
s = np.ones_like(theta)
l = np.ones_like(theta) / 2

r, g, b = utils.HSLtoRGB(theta, s, l)

where = x * x + y * y > 1
RGB = np.zeros([rows, cols, 3], dtype=float)
RGB[:, :, 0] = r.reshape([rows, cols])
RGB[:, :, 1] = g.reshape([rows, cols])
RGB[:, :, 2] = b.reshape([rows, cols])
RGB[:, :, 0][where] = 1
RGB[:, :, 1][where] = 1
RGB[:, :, 2][where] = 1
img = ax.imshow(RGB, extent=[-2, 2, -2, 2], origin="lower", vmin=-1, vmax=1)

ax.spines['left'].set_position('center')
ax.spines['bottom'].set_position('center')

ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')

ax.xaxis.set_ticks_position('bottom')
ax.yaxis.set_ticks_position('left')
ax.set_xlim([-1.0, 1.0])
ax.set_ylim([-1.0, 1.0])
ax.set_xticks([-1, 1])
ax.set_yticks([-1, 1])

ax.text(0, 1.1, "m$\\mathsf{_y}$", verticalalignment='center', horizontalalignment='center')
ax.text(1.1, 0, "m$\\mathsf{_x}$", verticalalignment='center', horizontalalignment='center')

plt.show()
fig.savefig("./HSL.png", facecolor="white", bbox_inches="tight")
