import numpy as np
from scipy.special import gamma as tgamma
from scipy.special import gammaln
from numpy import exp, log, sqrt, cos, sin
from numpy import abs as fabs
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


M_PI = np.pi

def normal_distribution(n: int) -> np.ndarray:
    u1 = np.random.random(n)
    u2 = np.random.random(n)
    return sqrt(-2.0 * log(u1)) * cos(2.0 * M_PI * u2)

def true_random_gsa(x: np.ndarray, qV: float, T: float, D: int) -> np.ndarray:
    f1 = ((qV - 1.0) / M_PI) ** (D / 2.0)
    f2 = tgamma(1.0 / (qV - 1.0) + (D - 1.0) / 2.0) / tgamma(1.0 / (qV - 1.0) - 1.0 / 2.0)
    f3 = T ** (-D / (3.0 - qV))
    f4 = 1.0 + (qV - 1.0) * x * x / (T ** (2.0 / (3.0 - qV)))
    f4 = f4 ** (1.0 / (qV - 1.0) + (D - 1.0) / 2.0)
    return f1 * f2 * f3 / f4

def get_random_gsa(qV: float, T: float, n: int, D: int = 1) -> np.ndarray:
    if D == 1:
        f1 = exp(log(T) / (qV - 1.0));
        f2 = exp((4.0 - qV) * log(qV - 1.0));
        f3 = exp((2.0 - qV) * log(2.0) / (qV - 1.0));
        f4 = sqrt(M_PI) * f1 * f2 / (f3 * (3.0 - qV));
        f5 = 1.0 / (qV - 1.0) - 0.5;
        f6 = M_PI * (1.0 - f5) / sin(M_PI * (1.0 - f5)) / tgamma(2.0 - f5);
        sigmax = exp(-(qV - 1.0) * log(f6 / f4) / (3.0 - qV));
        x = sigmax * normal_distribution(n)
        y = normal_distribution(n);
        den = exp((qV - 1.0) * log(fabs(y)) / (3.0 - qV));
        return x / den
    xs = np.random.random(n)
    x = 1.0 / ((1.0 + (1.0 - qV) * xs * xs / (T ** (2.0 / (3.0 - qV)))) ** (1.0 / (1.0 - qV) - 0.5))
    signal = np.random.random(n)
    return x * ((-1) * (signal < 0.5) + (signal > 0.5))



def plot_for_param(axl, axr, qV: float, T: float, D: int, n: int, minv: float = -5, maxv: float = 5):
    my_gsa = get_random_gsa(qV, T, n, D)
    
    #my_gsa_start = my_gsa > minv
    #my_gsa = my_gsa[my_gsa_start]
    #
    #my_gsa_end = my_gsa < maxv
    #my_gsa = my_gsa[my_gsa_end]
    #
    #end, start = max(my_gsa), min(my_gsa)
    
    x = np.linspace(minv, maxv, n)
    
    true_gsa = true_random_gsa(x, qV, T, D)
    true_gsa = true_gsa
    bin_size = 0.1
    bins = np.arange(minv, maxv + 1, bin_size)
    axl.hist(my_gsa, bins=bins, density=True)
    
    axl.set_xlim((minv, maxv))
    axl.set_xlabel("$\\Delta x$")
    axl.set_ylabel("Histogram")
    axl.set_yticks([])
    
    axr.plot(x, true_gsa)
    
    axr.set_xlim((minv, maxv))
    axr.set_xlabel("$\\Delta x$")
    axr.set_ylabel("$g_{q_V}(\\Delta x)$")
    axr.set_yticks([])


n = 1000000

T0 = 0.1
dT = 0.5

qV = 2.5
D = 1

FixPlot(16, 16)
rows = 5
fig, ax = plt.subplots(ncols=2, nrows=rows)

for row in range(rows):
    plot_for_param(ax[row][0], ax[row][1], qV, T0 + row * dT, D, n)


fig.savefig("fig_test.png", facecolor="white", bbox_inches="tight", dpi=275)

#x = np.linspace(1, 2, n)
#plt.plot(x, log(tgamma(x)))
#plt.scatter(x, gammaln(x))
#plt.show()
