import utils
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle

utils.FixPlot(8, 8)
cmd = utils.CMDArgs("dummy", "dummy")
frames, gi, gp, raw = utils.ReadAnimationBinary("integrate_evolution.dat")
utils.CreateAnimationFromFrames(".", "motion.mp4", cmd, gi, gp)

data = pd.read_csv("./clusters.dat", header=None).fillna(-1)
xs = data.iloc[:, 1::3].to_numpy() - 0.25e-9
ys = data.iloc[:, 2::3].to_numpy() - 0.25e-9
size = data.iloc[:, 3::3].to_numpy()

fig, ax = plt.subplots()


min_x, max_x = -0.25e-9, 63 * 0.5e-9 + 0.25e-9
min_y, max_y = -0.25e-9, 63 * 0.5e-9 + 0.25e-9
sx, sy = max_x - min_x, max_y - min_y
x = np.linspace(min_x, max_x, 100)
y = np.linspace(min_y, max_y, 100)
x, y = np.meshgrid(x, y)
a = 1

z = np.sin(2.0 * np.pi * x / sx * a) + 0.25 * np.sin(4.0 * np.pi * x * a / sx)
z = (z - z.min()) / (z.max() - z.min())
ax.imshow(z, cmap="gray", extent=(min_x, max_x, min_y, max_y), origin="lower", interpolation="bicubic")

ax.scatter(xs, ys, s=0.5, color="blue")

ax.set_xlim((min_x, max_x))
ax.set_ylim((min_y, max_y))

ax.set_xticklabels((f"{i/1e-9:.0f}" for i in ax.get_xticks()))
ax.set_yticklabels((f"{i/1e-9:.0f}" for i in ax.get_yticks()))
ax.set_xlabel("$x$(nm)")
ax.set_ylabel("$y$(nm)")


fig.savefig("./trajectories.png", dpi=cmd.DPI, facecolor="white", bbox_inches="tight")

data = pd.read_csv("integrate_info.dat")
Ex = data["eletric_x(V/m)"].to_numpy()
Ey = data["eletric_y(V/m)"].to_numpy()
Bz = data["magnetic_lattice_z(T)"].to_numpy() * 0.5 + data["magnetic_derivative_z(T)"].to_numpy() * 0.5
vx = Ey / Bz
vy = -Ex / Bz

