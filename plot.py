import utils
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import pandas as pd

from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle

def fix_cluster(input_dir, output_dir, dx, dy, cut, size):
    import ctypes
    _structure = ctypes.CDLL("/home/jose/.local/lib/atomistic/libatomistic.so")
    _structure.organize_clusters.argtypes = (ctypes.c_char_p, ctypes.c_char_p, ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_bool)
    input_dir = input_dir + '\0'
    output_dir = output_dir + '\0'
    input_dir = ctypes.create_string_buffer(input_dir.encode("ascii"))
    output_dir = ctypes.create_string_buffer(output_dir.encode("ascii"))
    _structure.organize_clusters(input_dir, output_dir, ctypes.c_double(dx), ctypes.c_double(dy), ctypes.c_double(cut), ctypes.c_bool(size))

utils.FixPlot(8, 8)
cmd = utils.CMDArgs("dummy", "dummy")
frames, gi, gp, raw = utils.ReadAnimationBinary("integrate_evolution.dat")
#utils.CreateAnimationFromFrames(".", "motion.mp4", cmd, gi, gp)

fix_cluster("./clusters.dat", "./clusters_org.dat", gi.cols * gp[0].lattice, gi.rows * gp[0].lattice, 1e8, 1)
data = pd.read_csv("./clusters_org.dat", header=None).fillna(-1)
xs = data.iloc[:, 1::3].to_numpy() - 0.25e-9
ys = data.iloc[:, 2::3].to_numpy() - 0.25e-9
size = data.iloc[:, 3::3].to_numpy()

fig, ax = plt.subplots()


min_x, max_x = -0.25e-9, gi.cols * 0.5e-9 + 0.25e-9
min_y, max_y = -0.25e-9, gi.rows * 0.5e-9 + 0.25e-9
sx, sy = max_x - min_x, max_y - min_y
x = np.linspace(min_x, max_x, 100)
y = np.linspace(min_y, max_y, 100)
x, y = np.meshgrid(x, y)
a = 1

min_ani, max_ani = min(gp, key=lambda x: x.ani.ani).ani.ani, max(gp, key=lambda x: x.ani.ani).ani.ani
anis = np.zeros((gi.rows, gi.cols))
for i in range(gi.rows):
    for j in range(gi.cols):
        anis[i, j] = (gp[i * gi.cols + j].ani.ani - min_ani) / (max_ani - min_ani)



ani_cmap = LinearSegmentedColormap.from_list("mcmp", ["#ffffff00", "#000000bb"])
ax.imshow(anis.reshape((gi.rows, gi.cols)), cmap=ani_cmap, origin="lower", extent=[min_x, max_x, min_y, max_y], interpolation=cmd.INTERPOLATION)


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

