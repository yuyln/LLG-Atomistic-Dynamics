import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from Plooter import *
FixPlot(8, 8)
NANO = 1.0e-9

vels = pd.read_table("./output/anim_velocity.out", header=None, skiprows=1)
pos = pd.read_table("./output/anim_pos_xy.out", header=None, skiprows=1)
dt = pos[0][1] - pos[0][0]

vxs = np.diff(pos[1]) / dt
vys = np.diff(pos[2]) / dt

vxs = np.concatenate([vxs, [vxs[-1]]])
vys = np.concatenate([vys, [vys[-1]]])

fig, ax = plt.subplots()
ax.plot(vels[0] / NANO, vels[1], color="black", label="P")
ax.plot(vels[0] / NANO, vels[2], color="red", label="P")

ax.plot(vels[0] / NANO, vels[3], color="blue", label="I")
ax.plot(vels[0] / NANO, vels[4], color="green", label="I")

ax.plot(vels[0] / NANO, vxs, color="orange", label="D")
ax.plot(vels[0] / NANO, vys, color="pink", label="D")

ax.legend()

plt.show()