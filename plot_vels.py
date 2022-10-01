import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from Plooter import *

vels_nagaosa = pd.read_table("./output/anim_velocity.out", header=None)
pos = pd.read_table("./output/anim_cm_charge.out", header=None)

t = pos[0]
x = pos[1]
y = pos[2]
dxdt = np.array([])
dydt = np.array([])
for i in range(1, len(t) - 1):
    dxdt = np.concatenate([dxdt, [0.5 * (x[i + 1] - x[i - 1]) / (t[i + 1] - t[i - 1])]])
    dydt = np.concatenate([dydt, [0.5 * (y[i + 1] - y[i - 1]) / (t[i + 1] - t[i - 1])]])


FixPlot(8, 8)
fig, ax = plt.subplots()
ax.plot(t[1: len(t) - 1] / 1.0e-9, dxdt, color="black", label="$v_x$")
ax.plot(t[1: len(t) - 1] / 1.0e-9, dydt, "--", color="black", label="$v_y$")

ax.plot(vels_nagaosa[0] / 1.0e-9, vels_nagaosa[1], color="red", label=r"$v_x^\mathrm{Nagaosa}$")
ax.plot(vels_nagaosa[0] / 1.0e-9, vels_nagaosa[2], "--", color="red", label=r"$v_y^\mathrm{Nagaosa}$")

ax.legend()
ax.set_xlabel("$t\mathrm{(ns)}$")
ax.set_ylabel("$v_x, v_y\mathrm{(m/s)}$")
plt.show()
fig.savefig("./imgs/out_vel.png", dpi=500, facecolor="white", bbox_inches='tight')

