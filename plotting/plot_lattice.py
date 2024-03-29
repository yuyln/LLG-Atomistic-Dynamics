import utils
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
import pandas as pd

cmd_parser = utils.CMDArgs("./output/end.bin", "./imgs/end_lattice.png")

rows, cols, lattice, data = utils.ReadLatticeBinary(cmd_parser.INPUT_FILE)
lattice = lattice / 1.0e-9

mx, my, mz = utils.GetFrameFromBinary(rows, cols, 1, data, 0, 16)
mz *= cmd_parser.INVERT
x, y, x_in, y_in, fac_x, fac_y = utils.GetPosition(rows, cols, cmd_parser.REDUCE_FACTOR, lattice)

min_x = -lattice / 2
min_y = -lattice / 2
max_x = (cols - 1) * lattice + lattice / 2
max_y = (rows - 1) * lattice + lattice / 2

if cmd_parser.USE_LATEX: utils.FixPlot(cmd_parser.WIDTH, cmd_parser.HEIGHT)
else: utils.FixPlot_(cmd_parser.WIDTH, cmd_parser.HEIGHT)

r = cols / rows
fig = plt.figure()

fig.set_size_inches(cmd_parser.WIDTH, cmd_parser.HEIGHT)
ax = fig.add_axes([0.12, 0.12, 0.75, 0.83])


if cmd_parser.HSL:
    rh, gh, bh = utils.GetHSL(mx, my, mz)
    RGB = np.zeros([rows, cols, 3], dtype=float)
    RGB[:, :, 0] = rh.reshape([rows, cols])
    RGB[:, :, 1] = gh.reshape([rows, cols])
    RGB[:, :, 2] = bh.reshape([rows, cols])
    img = ax.imshow(RGB, extent=[min_x, max_x, min_y, max_y], origin="lower", vmin=-1, vmax=1, interpolation=cmd_parser.INTERPOLATION)
else:
    global bar
    img = ax.imshow(mz.reshape([rows, cols]), cmap=utils.cmap, extent=[min_x, max_x, min_y, max_y], origin="lower", vmin=-1, vmax=1, interpolation=cmd_parser.INTERPOLATION)

    a = ax.get_position()
    x0, y0, x1, y1 = a.x0, a.y0, a.x1, a.y1

    pad = cmd_parser.BARPAD
    s = cmd_parser.BARSIZE

    if cmd_parser.BARPOS != "NONE":
        cax = None
        if cmd_parser.BARPOS.to_lower() == "top":
            cax = fig.add_axes([x0, y1 + 0.5 * s, x1 - x0, s])
        else:
            cax = fig.add_axes([x1 + 0.5 * s, y0, s, y1 - y0])

        bar = plt.colorbar(img, cax=cax, pad=pad, location=cmd_parser.BARPOS)
    elif cols > rows:
        cax = fig.add_axes([x0, y1 + 0.5 * s, x1 - x0, s])
        bar = plt.colorbar(img, cax=cax, pad=pad, location="top", orientation="horizontal")
        bar.set_ticks([-1, 0, 1])
    else:
        cax = fig.add_axes([x1 + 0.5 * s, y0, s, y1 - y0])
        bar = plt.colorbar(img, cax=cax, pad=pad, location="right")
        bar.set_ticks([-1, 0, 1])

#    divider = make_axes_locatable(ax)
#    cax1 = divider.append_axes("right", size="5%", pad=0.05)
#    bar = plt.colorbar(img, cax=cax1)
#    bar.set_ticks([-1, 0, 1])



if cmd_parser.PLOT_ARROWS:
    mx_, my_ = utils.GetVecsFromXY(mx, my, x_in, y_in)
    vecs = ax.quiver(x, y, mx_ * fac_x, my_ * fac_y,
                     angles='xy', scale_units='xy', pivot="mid", scale=1.0 / np.sqrt(1.0), width=0.0026, headwidth=3)

ax.set_xlim([min_x, max_x])
ax.set_ylim([min_y, max_y])

if cmd_parser.USE_LATEX:
    ax.set_xlabel("$x$(nm)")
    ax.set_ylabel("$y$(nm)")
    if not cmd_parser.HSL:
        bar.set_label("$m_z$")
else:
    ax.set_xlabel("x(nm)")
    ax.set_ylabel("y(nm)")
    if not cmd_parser.HSL:
        bar.set_label("m$\\mathsf{_z}$")

ax.text(-1, (rows - 1) * lattice, cmd_parser.LABEL,
        fontsize=28, verticalalignment='center', horizontalalignment='center')

if cmd_parser.PLOT_ANI:
    try:
        ani = pd.read_table(cmd_parser.ANI_INPUT, header=None, sep=cmd_parser.ANI_SEP, skiprows=cmd_parser.ANI_SKIP)
        y_ani = ani[0].to_numpy() * lattice
        x_ani = ani[1].to_numpy() * lattice
    except Exception as e:
        y_ani = np.array([]) * lattice
        x_ani = np.array([]) * lattice
        print(e)
    import matplotlib.collections as cl
    squares = [plt.Rectangle((xi - lattice / 2.0, yi - lattice / 2.0), lattice, lattice) for xi,yi in zip(x_ani,y_ani)]
    s = cl.PatchCollection(squares, facecolor=cmd_parser.COLOR_ANI, edgecolor=None, linewidth=0)
    ax.add_collection(s)

if cmd_parser.PLOT_PIN:
    try:
        pin = pd.read_table(cmd_parser.PIN_INPUT, header=None, sep=cmd_parser.PIN_SEP, skiprows=cmd_parser.PIN_SKIP)
        y_pin = pin[0].to_numpy() * lattice
        x_pin = pin[1].to_numpy() * lattice
    except Exception as e:
        y_pin = np.array([]) * lattice
        x_pin = np.array([]) * lattice
        print(e)
    import matplotlib.collections as cl
    squares = [plt.Rectangle((xi - lattice / 2.0, yi - lattice / 2.0), lattice, lattice) for xi,yi in zip(x_pin,y_pin)]
    s = cl.PatchCollection(squares, facecolor=cmd_parser.COLOR_PIN, edgecolor=None, linewidth=0)
    ax.add_collection(s)

ax.plot([0, lattice / 2], [10 * lattice, 10 * lattice])
cmd_parser.print()
fig.savefig(cmd_parser.OUTPUT_FILE, dpi=cmd_parser.DPI, facecolor="white", bbox_inches="tight")
