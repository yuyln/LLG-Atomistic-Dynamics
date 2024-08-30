import utils
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

utils.FixPlot(8, 8)
cmd = utils.CMDArgs("dummy", "dummy")
frames, gi, gp, raw = utils.ReadAnimationBinary("integrate_evolution.dat")
data = pd.read_csv("./integrate_info.dat")
time, x, y, z, _, _, _ = utils.GetClusterData("./")

mx, my, mz = utils.GetFrameFromBinary(frames, gi, raw, frames - 1)
mx = mx.reshape((gi.depth, gi.rows, gi.cols))
my = my.reshape((gi.depth, gi.rows, gi.cols))
mz = mz.reshape((gi.depth, gi.rows, gi.cols))
r, g, b = utils.GetHSL(mx, my, mz)

with open("test.vtk", "w") as f:
    f.write("# vtk DataFile Version 2.0\n")
    f.write("Lattice\n")
    f.write("ASCII\n")

    f.write("DATASET STRUCTURED_GRID\n")
    f.write(f"DIMENSIONS {gi.cols} {gi.rows} {gi.depth}\n")

    f.write(f"POINTS {gi.rows * gi.cols * gi.depth} float\n")
    for z in range(gi.depth):
        for y in range(gi.rows):
            for x in range(gi.cols):
                f.write(f"{float(x):.1f} {float(y):.1f} {float(z):.1f}\n")
    f.write("\n")

    mx, my, mz = utils.GetFrameFromBinary(frames, gi, raw, frames - 1)
    mx = mx.reshape((gi.depth, gi.rows, gi.cols))
    my = my.reshape((gi.depth, gi.rows, gi.cols))
    mz = mz.reshape((gi.depth, gi.rows, gi.cols))
    r, g, b = utils.GetHSL(mx, my, mz)

    f.write(f"POINT_DATA {gi.rows * gi.cols * gi.depth}\n")
    f.write("VECTORS spins float\n")
    for z in range(gi.depth):
        for y in range(gi.rows):
            for x in range(gi.cols):
                f.write(f"{mx[z, y, x]:.4f} {my[z, y, x]:.4f} {mz[z, y, x]:.4f}\n")
    f.write("\n")

    f.write("SCALARS colors_idx float 1\n")
    f.write("LOOKUP_TABLE colors_table\n")
    for x in range(gi.cols * gi.rows * gi.depth):
            f.write(f"{(x) / (gi.rows * gi.cols * gi.depth - 1)} ")
    f.write("\n")

    f.write(f"LOOKUP_TABLE colors_table {gi.rows * gi.cols * gi.depth}\n")
    for z in range(gi.depth):
        for y in range(gi.rows):
            for x in range(gi.cols):
                f.write(f"{r[z, y, x]} {g[z, y, x]} {b[z, y, x]} 1\n")

    f.write(f"FIELD mz_field 1\n")
    f.write(f"mz_table 1 {gi.rows * gi.cols * gi.depth} float\n")
    for z in range(gi.depth):
        for y in range(gi.rows):
            for x in range(gi.cols):
                f.write(f"{mz[z, y, x]}\n")
    f.write("TEXTURE_COORDINATES tcoords 2 float\n")
    for z in range(gi.depth):
        for y in range(gi.rows):
            for x in range(gi.cols):
                angle = np.arctan2(my[z, y, x], mx[z, y, x])
                angle += np.pi
                angle /= 2.0 * np.pi
                light = (mz[z, y, x] + 1) * 0.5
                f.write(f"{angle} {light}\n")

exit(1)
with open("test.vtk", "w") as f:
    f.write("# vtk DataFile Version 2.0\n")
    f.write("Lattice\n")
    f.write("ASCII\n")

    f.write("DATASET UNSTRUCTURED_GRID\n")

    f.write(f"POINTS {(gi.rows + 1) * (gi.cols + 1) * (gi.depth + 1)} float\n")
    for z in range(gi.depth + 1):
        for y in range(gi.rows + 1):
            for x in range(gi.cols + 1):
                f.write(f"{float(x):.1f} {float(y):.1f} {float(z):.1f}\n")

    f.write("\n")

    f.write(f"CELLS {gi.rows * gi.cols * gi.depth} {9 * gi.rows * gi.cols * gi.depth}\n")
    def idx(x, y, z):
        return z * (gi.rows + 1) * (gi.cols + 1) + y * (gi.cols + 1) + x
    for z in range(gi.depth):
        for y in range(gi.rows):
            for x in range(gi.cols):
                f.write(f"8 {idx(x, y, z)} {idx(x, y + 1, z)} {idx(x, y, z + 1)} {idx(x, y + 1, z + 1)} ")
                f.write(f"{idx(x + 1, y, z)} {idx(x + 1, y + 1, z)} {idx(x + 1, y, z + 1)} {idx(x + 1, y + 1, z + 1)}\n")
    f.write("\n")
    f.write(f"CELL_TYPES {gi.rows * gi.cols * gi.depth}\n")
    for z in range(gi.depth):
        for y in range(gi.rows):
            for x in range(gi.cols):
                f.write(f"11\n")
    f.write("\n")

    f.write(f"CELL_DATA {gi.rows * gi.cols * gi.depth}\n")
    f.write("VECTORS spins float\n")
    for z in range(gi.depth):
        for y in range(gi.rows):
            for x in range(gi.cols):
                f.write(f"{mx[z, y, x]:.4f} {my[z, y, x]:.4f} {mz[z, y, x]:.4f}\n")
    f.write("\n")

    f.write("SCALARS colors_idx float 1\n")
    f.write("LOOKUP_TABLE colors_table\n")
    for x in range(gi.cols * gi.rows * gi.depth):
            f.write(f"{(x) / (gi.rows * gi.cols * gi.depth - 1)} ")
    f.write("\n")

    f.write(f"LOOKUP_TABLE colors_table {gi.rows * gi.cols * gi.depth}\n")
    for z in range(gi.depth):
        for y in range(gi.rows):
            for x in range(gi.cols):
                f.write(f"{r[z, y, x]} {g[z, y, x]} {b[z, y, x]} 1\n")

    f.write(f"FIELD mz_field 1\n")
    f.write(f"mz_table 1 {gi.rows * gi.cols * gi.depth} float\n")
    for z in range(gi.depth):
        for y in range(gi.rows):
            for x in range(gi.cols):
                f.write(f"{mz[z, y, x]}\n")
exit(1)

fig = plt.figure()
ax = fig.add_subplot(projection='3d')

ax.scatter(x, y, z)
ax.set_xlim((0, gi.cols * gi.lattice))
ax.set_ylim((0, gi.rows * gi.lattice))
ax.set_zlim((0, gi.depth * gi.lattice))
plt.show()

fig, ax = plt.subplots()

min_x, max_x = 0, gi.cols * gi.lattice
min_y, max_y = 0, gi.rows * gi.lattice
min_z, max_z = 0, gi.depth * gi.lattice
sx, sy, sz = max_x - min_x, max_y - min_y, max_z - min_z

mx, my, mz = utils.GetFrameFromBinary(frames, gi, raw, frames - 1)
mx = mx.reshape((gi.depth, gi.rows, gi.cols))
my = my.reshape((gi.depth, gi.rows, gi.cols))
mz = mz.reshape((gi.depth, gi.rows, gi.cols))

for i in range(gi.cols):
    mx_ = mx[:, :, i]
    my_ = my[:, :, i]
    mz_ = mz[:, :, i]

    r, g, b = utils.GetHSL(mx_, my_, mz_ * cmd.INVERT)
    rgb = np.zeros((*mx_.shape, 3))
    rgb[:, :, 0] = r.reshape((mx_.shape))
    rgb[:, :, 1] = g.reshape((mx_.shape))
    rgb[:, :, 2] = b.reshape((mx_.shape))
    plt.imshow(rgb, origin="lower", cmap=utils.cmap, vmax=1, vmin=-1)
    plt.show()


exit(1)
with open("test.vtk", "w") as f:
    f.write("# vtk DataFile Version 2.0\n")
    f.write("Lattice\n")
    f.write("ASCII\n")

    f.write("DATASET STRUCTURED_GRID\n")
    f.write(f"DIMENSIONS {gi.cols} {gi.rows} {gi.depth}\n")

    f.write(f"POINTS {gi.rows * gi.cols * gi.depth} float\n")
    for z in range(gi.depth):
        for y in range(gi.rows):
            for x in range(gi.cols):
                f.write(f"{float(x):.1f} {float(y):.1f} {float(z):.1f}\n")
    f.write("\n")

    mx, my, mz = utils.GetFrameFromBinary(frames, gi, raw, frames - 1)
    mx = mx.reshape((gi.depth, gi.rows, gi.cols))
    my = my.reshape((gi.depth, gi.rows, gi.cols))
    mz = mz.reshape((gi.depth, gi.rows, gi.cols))
    r, g, b = utils.GetHSL(mx, my, mz)

    f.write(f"POINT_DATA {gi.rows * gi.cols * gi.depth}\n")
    f.write("VECTORS spins float\n")
    for z in range(gi.depth):
        for y in range(gi.rows):
            for x in range(gi.cols):
                f.write(f"{mx[z, y, x]:.4f} {my[z, y, x]:.4f} {mz[z, y, x]:.4f}\n")
    f.write("\n")

    f.write("SCALARS colors_idx float 1\n")
    f.write("LOOKUP_TABLE colors_table\n")
    for x in range(gi.cols * gi.rows * gi.depth):
            f.write(f"{(x) / (gi.rows * gi.cols * gi.depth - 1)} ")
    f.write("\n")

    f.write(f"LOOKUP_TABLE colors_table {gi.rows * gi.cols * gi.depth}\n")
    for z in range(gi.depth):
        for y in range(gi.rows):
            for x in range(gi.cols):
                f.write(f"{r[z, y, x]} {g[z, y, x]} {b[z, y, x]} 1\n")

    f.write(f"FIELD mz_field 1\n")
    f.write(f"mz_table 1 {gi.rows * gi.cols * gi.depth} float\n")
    for z in range(gi.depth):
        for y in range(gi.rows):
            for x in range(gi.cols):
                f.write(f"{mz[z, y, x]}\n")

    f.write(f"FIELD color_helper 1\n")
    f.write(f"color_helper_t 1 {gi.rows * gi.cols * gi.depth} float\n")
    for z in range(gi.depth):
        for y in range(gi.rows):
            for x in range(gi.cols):
                angle = np.arctan2(my[z, y, x], mx[z, y, x])
                angle += np.pi
                if angle > np.pi:
                    angle = np.pi - (angle - np.pi)
                angle /= np.pi
                light = mz[z, y, x]
                light += 1
                light /= 2
                light += 1
                value = angle
                f.write(f"{value}\n")
exit(1)

fig = plt.figure()
ax = fig.add_subplot(projection='3d')

ax.scatter(x, y, z)
ax.set_xlim((0, gi.cols * gi.lattice))
ax.set_ylim((0, gi.rows * gi.lattice))
ax.set_zlim((0, gi.depth * gi.lattice))
plt.show()

fig, ax = plt.subplots()

min_x, max_x = 0, gi.cols * gi.lattice
min_y, max_y = 0, gi.rows * gi.lattice
min_z, max_z = 0, gi.depth * gi.lattice
sx, sy, sz = max_x - min_x, max_y - min_y, max_z - min_z

mx, my, mz = utils.GetFrameFromBinary(frames, gi, raw, frames - 1)
mx = mx.reshape((gi.depth, gi.rows, gi.cols))
my = my.reshape((gi.depth, gi.rows, gi.cols))
mz = mz.reshape((gi.depth, gi.rows, gi.cols))

for i in range(gi.cols):
    mx_ = mx[:, :, i]
    my_ = my[:, :, i]
    mz_ = mz[:, :, i]

    r, g, b = utils.GetHSL(mx_, my_, mz_ * cmd.INVERT)
    rgb = np.zeros((*mx_.shape, 3))
    rgb[:, :, 0] = r.reshape((mx_.shape))
    rgb[:, :, 1] = g.reshape((mx_.shape))
    rgb[:, :, 2] = b.reshape((mx_.shape))
    plt.imshow(rgb, origin="lower", cmap=utils.cmap, vmax=1, vmin=-1)
    plt.show()

