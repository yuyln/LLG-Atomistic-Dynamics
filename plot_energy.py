import utils
from Plooter import FixPlot, FixPlot_
import matplotlib.pyplot as plt

cmd_parser = utils.CMDArgs("./output/anim_energy.out", "./imgs/plot_energy.png")

data = utils.ReadFile(cmd_parser.INPUT_FILE)

if cmd_parser.USE_LATEX: FixPlot(cmd_parser.WIDTH, cmd_parser.HEIGHT)
else: FixPlot_(cmd_parser.WIDTH, cmd_parser.HEIGHT)

fig, ax = plt.subplots()
fig.set_size_inches(cmd_parser.WIDTH, cmd_parser.HEIGHT)

ax.plot(data[0] / utils.NANO, data[1] / utils.QE)

if cmd_parser.USE_LATEX:
    ax.set_xlabel("$t$(ns)")
    ax.set_ylabel("$E$(eV)")
else:
    ax.set_xlabel("t(ns)")
    ax.set_ylabel("E(eV)")

ax.set_xlim([min(data[0]) / utils.NANO, max(data[0]) / utils.NANO])

cmd_parser.print()
fig.savefig(cmd_parser.OUTPUT_FILE, dpi=cmd_parser.DPI, facecolor="white", bbox_inches="tight")
