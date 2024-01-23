import numpy as np
import matplotlib.pyplot as plt

def FixPlot(lx: float, ly: float):
    from matplotlib import rcParams, cycler
    rcParams['font.family'] = 'serif'
    rcParams['font.serif'] = ['Computer Modern']
    rcParams['text.usetex'] = True
    rcParams['font.size'] = 28
    rcParams['axes.linewidth'] = 1.1
    rcParams['axes.labelpad'] = 10.0
    plot_color_cycle = cycler('color', ['000000', 'FE0000', '0000FE', '008001', 'FD8000', '8c564b',
                                        'e377c2', '7f7f7f', 'bcbd22', '17becf'])
    rcParams['axes.prop_cycle'] = plot_color_cycle
    rcParams['axes.xmargin'] = 0
    rcParams['axes.ymargin'] = 0
    rcParams['legend.fancybox'] = False
    rcParams['legend.framealpha'] = 1.0
    rcParams['legend.edgecolor'] = "black"
    rcParams['legend.fontsize'] = 22
    rcParams['xtick.labelsize'] = 22
    rcParams['ytick.labelsize'] = 22

    rcParams['ytick.right'] = True
    rcParams['xtick.top'] = True

    rcParams['xtick.direction'] = "in"
    rcParams['ytick.direction'] = "in"
    rcParams['axes.formatter.useoffset'] = False

    rcParams.update({"figure.figsize": (lx, ly),
                    "figure.subplot.left": 0.177, "figure.subplot.right": 0.946,
                     "figure.subplot.bottom": 0.156, "figure.subplot.top": 0.965,
                     "axes.autolimit_mode": "round_numbers",
                     "xtick.major.size": 7,
                     "xtick.minor.size": 3.5,
                     "xtick.major.width": 1.1,
                     "xtick.minor.width": 1.1,
                     "xtick.major.pad": 5,
                     "xtick.minor.visible": True,
                     "ytick.major.size": 7,
                     "ytick.minor.size": 3.5,
                     "ytick.major.width": 1.1,
                     "ytick.minor.width": 1.1,
                     "ytick.major.pad": 5,
                     "ytick.minor.visible": True,
                     "lines.markersize": 10,
                     "lines.markeredgewidth": 0.8, 
                     "mathtext.fontset": "cm"})
def get_dm(ri: tuple[float, float, float], rj: tuple[float, float, float]) -> tuple[np.ndarray, np.ndarray]:
    ri_ = np.array(ri)
    rj_ = np.array(rj);
    rij = rj_ - ri_
    return rij / 2, np.cross(rij, (0, 0, 1))
    return rij / 2, rij
    return rij / 2, np.cross(rij, (0, 1, 0.5))

FixPlot(8, 8)
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter((0), (0), s=50)
for theta in [0.0, np.pi / 2.0, np.pi, 3.0 * np.pi / 2.0]:
    pos, uv = get_dm((0, 0, 0), (np.cos(theta), np.sin(theta), 0))
    ax.quiver(*pos, *uv, pivot="middle")
    ax.scatter((np.cos(theta)), (np.sin(theta)), s=50)

#pos, uv = get_dm((0, 0, 0), (0.0, 0.0, 1.0))
#ax.quiver(*pos, *uv, pivot="middle")
#ax.scatter((0.0), (0.0), (1.0), s=50)
#
#pos, uv = get_dm((0, 0, 0), (0.0, 0.0, -1.0))
#ax.quiver(*pos, *uv, pivot="middle")
#ax.scatter((0.0), (0.0), (-1.0), s=50)

ax.set_xlim((-2, 2))
ax.set_ylim((-2, 2))
ax.set_zlim((-2, 2))
plt.show()
