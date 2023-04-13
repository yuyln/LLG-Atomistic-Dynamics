import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from Plooter import *

data = pd.read_table("./output/anim_energy.out", header=None, sep="\t")

FixPlot(8, 8)
fig, ax = plt.subplots()
ax.plot(data[0] / 1.0e-9, data[1] / 1.6e-19)
ax.set_xlim([min(data[0]/1.0e-9), max(data[0]/1.0e-9)])
fig.savefig("./imgs/energy.png", dpi=250, facecolor="white", bbox_inches="tight")
