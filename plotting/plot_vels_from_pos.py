import utils
import matplotlib.pyplot as plt
import numpy as np

cmd_parser = utils.CMDArgs("./output/anim_pos_xy.out", "./imgs/test_velocities.png")

data = utils.ReadFile(cmd_parser.INPUT_FILE)
dt = data[0][1] - data[0][0]
vx = np.diff(data[1].to_numpy()) / dt
vy = np.diff(data[2].to_numpy()) / dt

if cmd_parser.USE_LATEX: utils.FixPlot(cmd_parser.WIDTH, cmd_parser.HEIGHT)
else: utils.FixPlot_(cmd_parser.WIDTH, cmd_parser.HEIGHT)

fig, ax = plt.subplots()
fig.set_size_inches(cmd_parser.WIDTH, cmd_parser.HEIGHT)

ax.plot(data[0][5:] / utils.NANO, vx[4:])
ax.plot(data[0][5:] / utils.NANO, vy[4:])

theta = np.arctan2(vy[4:], vx[4:]) * 180 / np.pi
#ax.plot(data[0][5:] / utils.NANO, theta)
print(theta)

if cmd_parser.USE_LATEX:
    ax.set_xlabel("$t$(ns)")
    ax.set_ylabel("$v_x, v_y$(ms$^{-1}$)")
else:
    ax.set_xlabel("t(ns)")
    ax.set_ylabel("v$\\mathsf{_x}$, v$\\mathsf{_y}$(ms$^{-1}$)")

ax.set_xlim([min(data[0]) / utils.NANO, max(data[0]) / utils.NANO])

cmd_parser.print()
fig.savefig(cmd_parser.OUTPUT_FILE, dpi=cmd_parser.DPI, facecolor="white", bbox_inches="tight")
