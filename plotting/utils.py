import argparse
from struct import unpack
import array
import mmap
import pprint
import numpy as np
import pandas as pd
import sys
from matplotlib.colors import LinearSegmentedColormap
import colorsys

VEC_SIZE = 3 * 8 # BYTES
HBAR = 1.054571817e-34 # J*s
QE = 1.602176634e-19 # C
MU_B = 9.2740100783e-24 # J/T
MU_0 = 1.25663706212e-6 # N/A^2
KB = 1.380649e-23 # J/K
NANO = 1.0e-9
BYTEORDER = sys.byteorder
UBO = ">"
if BYTEORDER == "little": UBO = "<"

colors = ["#037fff", "white", "#f40501"]
cmap = LinearSegmentedColormap.from_list("mcmp", colors)

class CMDArgs:
    def __init__(self, inp, out):
        parser = argparse.ArgumentParser(description="Configure plots parameters via CLI")
        parser.add_argument("-input", default=inp, nargs="?", type=str)
        parser.add_argument("-output", default=out, nargs="?", type=str)
        parser.add_argument("-DPI", default=275, nargs="?", type=int)
        parser.add_argument("-arrows", action="store_true")
        parser.add_argument("-factor", default=1, nargs="?", type=int)
        parser.add_argument("-fps", default=60, nargs="?", type=int)
        parser.add_argument("-anisotropy", action="store_true")
        parser.add_argument("-pinning", action="store_true")
        parser.add_argument("-anisotropy-color", default="black", nargs="?", type=str)
        parser.add_argument("-pinning-color", default="yellow", nargs="?", type=str)
        parser.add_argument("-interpolation", default="nearest", nargs="?", type=str)
        parser.add_argument("-latex", action="store_true")
        parser.add_argument("-width", default=8, nargs="?", type=float)
        parser.add_argument("-height", default=8, nargs="?", type=float)
        parser.add_argument("-anisotropy-input", default="./input/anisotropy.in", nargs="?", type=str)
        parser.add_argument("-pinning-input", default="./input/pinning.in", nargs="?", type=str)
        parser.add_argument("-anisotropy-skiprows", default=2, nargs="?", type=int)
        parser.add_argument("-pinning-skiprows", default=2, nargs="?", type=int)
        parser.add_argument("-anisotropy-sep", default="\t", nargs="?", type=str)
        parser.add_argument("-pinning-sep", default="\t", nargs="?", type=str)
        parser.add_argument("-batch-size", default=50, nargs="?", type=int)
        parser.add_argument("-HSL", action="store_true")
        parser.add_argument("-invert", action="store_true")
        parser.add_argument("-bar-size", default=0.02, type=float)
        parser.add_argument("-bar-pad", default=0.00, type=float)
        parser.add_argument("-bar-pos", default="NONE", type=str)
        parser.add_argument("-label", default="", type=str)

        self.args = vars(parser.parse_args())
        self.INPUT_FILE    = self.args["input"]
        self.OUTPUT_FILE   = self.args["output"]
        self.DPI           = self.args["DPI"]
        self.PLOT_ARROWS   = self.args["arrows"]
        self.REDUCE_FACTOR = self.args["factor"]
        self.FPS           = self.args["fps"]
        self.PLOT_ANI      = self.args["anisotropy"]
        self.PLOT_PIN      = self.args["pinning"]
        self.COLOR_ANI     = self.args["anisotropy_color"]
        self.COLOR_PIN     = self.args["pinning_color"]
        self.INTERPOLATION = self.args["interpolation"]
        self.USE_LATEX     = self.args["latex"]
        self.WIDTH         = self.args["width"]
        self.HEIGHT        = self.args["height"]
        self.ANI_INPUT     = self.args["anisotropy_input"]
        self.PIN_INPUT     = self.args["pinning_input"]
        self.ANI_SKIP      = self.args["anisotropy_skiprows"]
        self.PIN_SKIP      = self.args["pinning_skiprows"]
        self.ANI_SEP       = self.args["anisotropy_sep"]
        self.PIN_SEP       = self.args["pinning_sep"]
        self.BATCH_S       = self.args["batch_size"]
        self.HSL           = self.args["HSL"]
        self.BARSIZE       = self.args["bar_size"]
        self.BARPAD        = self.args["bar_pad"]
        self.BARPOS        = self.args["bar_pos"]
        self.LABEL         = self.args["label"]
        self.INVERT        = -1 if self.args["invert"] else 1

    def print(self):
        pprint.pprint(self.args)

def ReadAnimationBinaryOLD(path: str, cut: int, dt: float, lattice: float) -> tuple[int, int, int, int, float, float, mmap.mmap]:
    file = open(path, "r")
    raw_data = mmap.mmap(file.fileno(), 0, access=mmap.ACCESS_READ)
    rows, cols, frames = unpack(f"{UBO}iii", raw_data[:12])
    file.close()
    return rows, cols, frames, cut, dt, lattice, raw_data

def ReadLatticeBinaryOLD(path: str, lattice: float) -> tuple[int, int, float, mmap.mmap]:
    file = open(path, "r")
    raw_data = mmap.mmap(file.fileno(), 0, access=mmap.ACCESS_READ)
    rows, cols = unpack(f"{UBO}ii", raw_data[:8])
    return rows, cols, lattice, raw_data

def ReadAnimationBinary(path: str) -> tuple[int, int, int, int, float, float, mmap.mmap]:
    file = open(path, "r")
    raw_data = mmap.mmap(file.fileno(), 0, access=mmap.ACCESS_READ)
    rows, cols, frames, cut, dt, lattice = unpack(f"{UBO}iiiidd", raw_data[:32])
    file.close()
    return rows, cols, frames, cut, dt, lattice, raw_data

def ReadLatticeBinary(path: str) -> tuple[int, int, float, mmap.mmap]:
    file = open(path, "r")
    raw_data = mmap.mmap(file.fileno(), 0, access=mmap.ACCESS_READ)
    rows, cols, lattice = unpack(f"{UBO}iid", raw_data[:16])
    return rows, cols, lattice, raw_data

def GetFrameFromBinary(rows: int, cols: int, frames: int, raw_data: mmap.mmap, i: int, offset: int=32) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if i < 0: i = 0
    elif i >= frames: i = frames - 1
    lat_s = rows * cols * VEC_SIZE
    raw_vecs = array.array("d")
    raw_vecs.frombytes(raw_data[offset + i * lat_s: offset + (i + 1) * lat_s])
    M = np.array(raw_vecs)
    mx, my, mz = M[0::3], M[1::3], M[2::3]
    return mx, my, mz

#def _v(m1, m2, hue):
#    hue = hue % 1.0
#    if hue < ONE_SIXTH:
#        return m1 + (m2-m1)*hue*6.0
#    if hue < 0.5:
#        return m2
#    if hue < TWO_THIRD:
#        return m1 + (m2-m1)*(TWO_THIRD-hue)*6.0
#    return m1

#def hls_to_rgb(h, l, s):
#    if s == 0.0:
#        return l, l, l
#    if l <= 0.5:
#        m2 = l * (1.0+s)
#    else:
#        m2 = l+s-(l*s)
#    m1 = 2.0*l - m2
#    return (_v(m1, m2, h+ONE_THIRD), _v(m1, m2, h), _v(m1, m2, h-ONE_THIRD))
#

def _v(m1, m2, hue):
    ret = np.ones_like(hue)
    hueL = np.ones_like(hue)
    ret[:] = hue[:]
    hueL = hue
    hueL = hueL % 1.0
 
    ret[:] = m1[:]
 
    cond = hueL < 2.0 / 3.0
    ret[cond] = m1[cond] + (m2[cond] - m1[cond]) * (2.0 / 3.0 - hueL[cond]) * 6.0

    cond = hueL < 0.5
    ret[cond] = m2[cond]

    cond = hueL < 1.0 / 6.0
    ret[cond] = m1[cond] + (m2[cond] - m1[cond]) * hueL[cond] * 6.0
 
    return ret 

def HSLtoRGB(h: np.ndarray, s: np.ndarray, l: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    r = np.empty_like(h)
    g = np.empty_like(h)
    b = np.empty_like(h)

    r[:] = l[:]
    g[:] = l[:]
    b[:] = l[:]

    m2 = l + s - (l * s)
    m2[l <= 0.5] = l[l <= 0.5] * (1.0 + s[l <= 0.5])

    m1 = 2.0 * l - m2

    r, g, b = _v(m1, m2, h + 1.0 / 3.0), _v(m1, m2, h), _v(m1, m2, h - 1.0 / 3.0)

    return r, g, b

def GetHSL(mx: np.ndarray, my: np.ndarray, mz: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    angle = np.arctan2(my, mx) / np.pi
    #angle += np.pi
    #angle *= 180 / np.pi
    angle = (angle + 1) / 2.0
    L = (mz + 1) / 2.0
    S = np.ones_like(L)
    return HSLtoRGB(angle, S, L)


def ReadFile(path: str, sep: str="\t") -> pd.DataFrame:
    return pd.read_table(path, header=None, sep=sep)


def GetPosition(rows: int, cols: int, reduce: int, lattice: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, float]:
    xs, ys = zip(*[[j, i] for i in range(rows) for j in range(cols)])
    xs = np.array(xs) * lattice
    ys = np.array(ys) * lattice
    xu = np.unique(xs)[::reduce]
    yu = np.unique(ys)[::reduce]

    x_in = np.in1d(xs, xu)

    y_in = ys[x_in]
    y_in = np.in1d(y_in, yu)

    x = xs[x_in]
    x = x[y_in]

    y = ys[x_in]
    y = y[y_in]

    facx = xu[1] - xu[0]
    facy = yu[1] - yu[0]

    return x, y, x_in, y_in, facx, facy

def GetVecsFromXY(mx_: np.ndarray, my_: np.ndarray, x_in: np.ndarray, y_in: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mx = mx_[x_in]
    mx = mx[y_in]

    my = my_[x_in]
    my = my[y_in]
    return mx, my

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

def FixPlot_(lx: float, ly: float):
    from matplotlib import rcParams, cycler
    rcParams['font.family'] = 'sans-serif'
    rcParams['font.sans-serif'] = ['Arial']
    rcParams['text.usetex'] = False
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
    rcParams['legend.fontsize'] = 28
    rcParams['xtick.labelsize'] = 22
    rcParams['ytick.labelsize'] = 22

    rcParams['ytick.right'] = True
    rcParams['xtick.top'] = True

    rcParams['xtick.direction'] = "in"
    rcParams['ytick.direction'] = "in"

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
                     "mathtext.fontset": "dejavusans"}) #"cm"

