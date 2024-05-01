import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv("./integrate_info.dat")
time = data["time(s)"].to_numpy()
Ex = data["eletric_x(V/m)"].to_numpy()
Ey = data["eletric_y(V/m)"].to_numpy()
Bz = data["magnetic_lattice_z(T)"].to_numpy() * 0.5 + data["magnetic_derivative_z(T)"].to_numpy() * 0.5
cx = data["charge_center_x(m)"].to_numpy()
cy = data["charge_center_y(m)"].to_numpy()
vx = Ey / Bz
vy = -Ex / Bz

fig, ax = plt.subplots()
ax.plot(time, vx, color="black")
ax.plot(time, vy, color="red")
plt.show()

fig, ax = plt.subplots()
ax.plot(time, np.arctan2(vy, vx) * 180.0 / np.pi, color="black")
plt.show()

fig, ax = plt.subplots()
ax.scatter(cx, cy, color="black")
plt.show()

