import utils
from Plooter import FixPlot, FixPlot_
import matplotlib.pyplot as plt

cmd_parser = utils.CMDArgs("./output/anim_velocity.out", "./imgs/plot_velocities.png")

data = utils.ReadFile(cmd_parser.INPUT_FILE)

if cmd_parser.USE_LATEX: FixPlot(cmd_parser.WIDTH, cmd_parser.HEIGHT)
else: FixPlot_(cmd_parser.WIDTH, cmd_parser.HEIGHT)

fig, ax = plt.subplots()
fig.set_size_inches(cmd_parser.WIDTH, cmd_parser.HEIGHT)

ax.plot(data[0] / utils.NANO, data[1])
ax.plot(data[0] / utils.NANO, data[2])

if cmd_parser.USE_LATEX:
    ax.set_xlabel("$t$(ns)")
    ax.set_ylabel("$v_x, v_y$(ms$^{-1}$)")
else:
    ax.set_xlabel("t(ns)")
    ax.set_ylabel("v$\\mathsf{_x}$, v$\\mathsf{_y}$(ms$^{-1}$)")

ax.set_xlim([min(data[0]) / utils.NANO, max(data[0]) / utils.NANO])

cmd_parser.print()
fig.savefig(cmd_parser.OUTPUT_FILE, dpi=cmd_parser.DPI, facecolor="white", bbox_inches="tight")
