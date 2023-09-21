import utils
import matplotlib.pyplot as plt

cmd_parser = utils.CMDArgs("./output/anim_charge.out", "./imgs/plot_charge.png")

data = utils.ReadFile(cmd_parser.INPUT_FILE)

if cmd_parser.USE_LATEX: utils.FixPlot(cmd_parser.WIDTH, cmd_parser.HEIGHT)
else: utils.FixPlot_(cmd_parser.WIDTH, cmd_parser.HEIGHT)

fig, ax = plt.subplots()
fig.set_size_inches(cmd_parser.WIDTH, cmd_parser.HEIGHT)

ax.plot(data[0] / utils.NANO, data[1])

if cmd_parser.USE_LATEX:
    ax.set_xlabel("$t$(ns)")
    ax.set_ylabel("$Q$")
else:
    ax.set_xlabel("t(ns)")
    ax.set_ylabel("Q")

ax.set_xlim([min(data[0]) / utils.NANO, max(data[0]) / utils.NANO])

cmd_parser.print()
fig.savefig(cmd_parser.OUTPUT_FILE, dpi=cmd_parser.DPI, facecolor="white", bbox_inches="tight")
