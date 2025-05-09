import argparse
from struct import unpack
import array
import mmap
import pprint
import numpy as np
import pandas as pd
import sys
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.pyplot as plt
import ctypes as ct
import numpy as np

VEC_SIZE = 3 * 8 # BYTES
HBAR = 1.054571817e-34 # J*s
QE = 1.602176634e-19 # C
MU_B = 9.2740100783e-24 # J/T
MU_0 = 1.25663706212e-6 # N/A^2
KB = 1.380649e-23 # J/K
NANO = 1.0e-9
BYTEORDER = sys.byteorder
UBO = ">"
EPS = 1.0e-30
if BYTEORDER == "little": UBO = "<"

def CLOSE_ENOUGH(a: float, b: float, eps: float = EPS) -> bool:
    return (a-b) ** 2.0 < eps ** 2.0

colors = ["#037fff", "white", "#f40501"]
colors = ["#0000ff", "white", "#ff0000"]
cmap = LinearSegmentedColormap.from_list("mcmp", colors)

class v3d(ct.Structure):
    _fields_ = [('x', ct.c_double), ('y', ct.c_double), ('z', ct.c_double)]
    def __str__(self):
        return f"({getattr(self, 'x')} {getattr(self, 'y')} {getattr(self, 'z')})"
    def __eq__(self, other):
        return CLOSE_ENOUGH(self.x, other.x) and CLOSE_ENOUGH(self.y, other.y) and CLOSE_ENOUGH(self.z, other.z)

class pbc_rules(ct.Structure):
    _fields_ = [('m', v3d), ('pbc_x', ct.c_int), ('pbc_y', ct.c_int), ('pbc_z', ct.c_int)]
    def __str__(self):
        s = "("
        for n, t in self._fields_:
            s += f"{n}: {getattr(self, n)} "
        s += ")"
        return s
    def __eq__(self, other):
        return self.pbc_x == other.pbc_x and self.pbc_y == other.pbc_y and self.pbc_z == other.pbc_z and self.m == other.m

class grid_info(ct.Structure):
    _fields_ = [('rows', ct.c_uint), ('cols', ct.c_uint), ('depth', ct.c_uint), ('lattice', ct.c_double), ('pbc', pbc_rules)]
    def __str__(self):
        s = "("
        for n, t in self._fields_:
            s += f"{n}: {getattr(self, n)} "
        s += ")"
        return s
    def __eq__(self, other):
            return self.rows == other.rows and self.cols == other.cols and self.depth == other.depth and CLOSE_ENOUGH(self.lattice, other.lattice) and self.pbc == other.pbc

class anisotropy(ct.Structure):
    _fields_ = [('dir', v3d), ('ani', ct.c_double)]
    def __str__(self):
        s = "("
        for n, t in self._fields_:
            s += f"{n}: {getattr(self, n)} "
        s += ")"
        return s
    def __eq__(self, other):
        return CLOSE_ENOUGH(self.ani, other.ani) and self.dir == other.dir

class pinning(ct.Structure):
    _fields_ = [('dir', v3d), ('pinned', ct.c_char)]
    def __str__(self):
        s = "("
        for n, t in self._fields_:
            s += f"{n}: {getattr(self, n)} "
        s += ")"
        return s
    def __eq__(self, other):
        return self.pinned == other.pinned and self.dir == other.dir

class dm_interaction(ct.Structure):
    _fields_ = [('dmv_left', v3d), ('dmv_right', v3d), ('dmv_up', v3d), ('dmv_down', v3d), ('dmv_front', v3d), ('dmv_back', v3d)]
    def __str__(self):
        s = "("
        for n, t in self._fields_:
            s += f"{n}: {getattr(self, n)}\n"
        s += ")"
        return s
    def __eq__(self, other):
        ret = True
        for n, t in self._fields_:
            ret = ret and getattr(self, n) == getattr(other, n)
        return ret

class exchange_interaction(ct.Structure):
    _fields_ = [('J_left', ct.c_double), ('J_right', ct.c_double), ('J_up', ct.c_double), ('J_down', ct.c_double), ('J_front', ct.c_double), ('J_back', ct.c_double)]
    def __str__(self):
        s = "("
        for n, t in self._fields_:
            s += f"{n}: {getattr(self, n)}\n"
        s += ")"
        return s
    def __eq__(self, other):
        ret = True
        for n, t in self._fields_:
            ret = ret and getattr(self, n) == getattr(other, n)
        return ret

class grid_site_params(ct.Structure):
    _fields_ = [('i', ct.c_int), ('j', ct.c_int), ('k', ct.c_int),
                ('cubic_ani', ct.c_double),
                ('mu', ct.c_double), ('alpha', ct.c_double), ('gamma', ct.c_double),
                ('ani', anisotropy), ('pin', pinning), ('dm', dm_interaction),
                ('exchange', exchange_interaction),
                ('lattice', ct.c_double)]
    def __str__(self):
        s = "gp: (\n"
        for n, t in self._fields_:
            s += f"{n}: {getattr(self, n)}\n"
        s += ")"
        return s
    def __eq__(self, other):
        ret = True
        for n, t in self._fields_:
            if n == "row" or n == "col":
                continue
            if t == ct.c_double:
                ret = ret and CLOSE_ENOUGH(getattr(self, n), getattr(other, n))
            else:
                ret = ret and getattr(self, n) == getattr(other, n)
        return ret

class CMDArgs:
    def __init__(self, *args):
        parser = argparse.ArgumentParser(description="Configure plots parameters via CLI")
        for param in args:
            stuff = args[param]
            parser.add_argument(param, default=stuff["default"],
                                       nargs=stuff["nargs"],
                                       action=stuff["action"],
                                       type=stuff["type"])
        parser.add_argument("-DPI", default=275, nargs="?", type=int)
        parser.add_argument("-arrows", action="store_true")
        parser.add_argument("-factor", default=1, nargs="?", type=int)
        parser.add_argument("-fps", default=60, nargs="?", type=int)
        parser.add_argument("-plot-defects", action="store_true")
        parser.add_argument("-interpolation", default="nearest", nargs="?", type=str)
        parser.add_argument("-latex", action="store_true")
        parser.add_argument("-width", default=8, nargs="?", type=float)
        parser.add_argument("-height", default=8, nargs="?", type=float)
        parser.add_argument("-HSL", action="store_true")
        parser.add_argument("-invert", action="store_true")
        parser.add_argument("-bar-size", default=0.02, type=float)
        parser.add_argument("-bar-pad", default=0.00, type=float)
        parser.add_argument("-bar-pos", default="NONE", type=str)
        parser.add_argument("-label", default="", type=str)

        self.args = vars(parser.parse_args())
        self.DPI           = self.args["DPI"]
        self.PLOT_ARROWS   = self.args["arrows"]
        self.REDUCE_FACTOR = self.args["factor"]
        self.FPS           = self.args["fps"]
        self.PLOT_DEF      = self.args["plot_defects"]
        self.INTERPOLATION = self.args["interpolation"]
        self.USE_LATEX     = self.args["latex"]
        self.WIDTH         = self.args["width"]
        self.HEIGHT        = self.args["height"]
        self.HSL           = self.args["HSL"]
        self.BARSIZE       = self.args["bar_size"]
        self.BARPAD        = self.args["bar_pad"]
        self.BARPOS        = self.args["bar_pos"]
        self.LABEL         = self.args["label"]
        self.INVERT        = -1 if self.args["invert"] else 1

    def print(self):
        pprint.pprint(self.args)

def ReadAnimationBinary(path: str) -> tuple[int, grid_info, list[grid_site_params], bytes]:
    file = open(path, "rb")
    class dummy(ct.Structure):
        _fields_ = [('frames', ct.c_uint64)]

    frames = dummy()
    file.readinto(frames)
    skip = ct.sizeof(frames)

    gi = grid_info()
    file.readinto(gi)
    skip += ct.sizeof(gi)

    gps = []
    for _ in range(gi.rows):
        for _ in range(gi.cols):
            for _ in range(gi.depth):
                gp = grid_site_params()
                file.readinto(gp)
                gps.append(gp)
    skip += ct.sizeof(gps[0]) * gi.rows * gi.cols * gi.depth
    file.close()
    file = open(path, "rb")

    raw_data = mmap.mmap(file.fileno(), 0, access=mmap.ACCESS_READ)[skip:]
    frames = len(raw_data) / (gi.rows * gi.cols * gi.depth * VEC_SIZE)

    file.close()
    return int(frames), gi, gps, raw_data

def ReadAnimationBinaryF(path: str) -> tuple[int, grid_info, list[grid_site_params]]:
    import os
    file = open(path, "rb")
    class dummy(ct.Structure):
        _fields_ = [('frames', ct.c_uint64)]

    frames = dummy()
    file.readinto(frames)
    skip = ct.sizeof(frames)

    gi = grid_info()
    file.readinto(gi)
    skip += ct.sizeof(gi)

    gps = []
    for _ in range(gi.rows):
        for _ in range(gi.cols):
            for _ in range(gi.depth):
                gp = grid_site_params()
                file.readinto(gp)
                gps.append(gp)
    skip += ct.sizeof(gps[0]) * gi.rows * gi.cols * gi.depth
    file.close()

    file_size = os.stat(path).st_size
    file = open(path, "rb")

    grid_size = file_size - skip
    frames = grid_size / (gi.rows * gi.cols * gi.depth * VEC_SIZE)

    return int(frames), gi, gps, file

def GetFrameFromBinaryF(frames: int, gi: grid_info, raw_file, i: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    i = ((i % frames) + frames) % frames
    lat_s = gi.rows * gi.cols * gi.depth * VEC_SIZE
    raw_file_skip = ct.sizeof(ct.uint64_t) + ct.sizeof(grid_info) + ct.sizeof(grid_site_params) * gi.rows * gi.cols * gi.depth + lat_s * i
    raw_file.skip(raw_file_skip, 0)
    raw_data = raw_file.read(lat_s)
    raw_vecs = array.array("d")
    raw_vecs.frombytes(raw_data)
    M = np.array(raw_vecs)
    mx, my, mz = M[0::3], M[1::3], M[2::3]
    return mx, my, mz

def ReadLatticeBinary(path: str) -> tuple[grid_info, list[grid_site_params], np.ndarray, np.ndarray, np.ndarray]:
    file = open(path, "rb")

    gi = grid_info()
    file.readinto(gi)

    gps = []
    for _ in range(gi.rows):
        for _ in range(gi.cols):
            for _ in range(gi.depth):
                gp = grid_site_params()
                file.readinto(gp)
                gps.append(gp)

    mx = np.zeros((gi.depth, gi.rows, gi.cols))
    my = np.zeros((gi.depth, gi.rows, gi.cols))
    mz = np.zeros((gi.depth, gi.rows, gi.cols))
    for k in range(gi.depth):
        for i in range(gi.rows):
            for j in range(gi.cols):
                m = v3d()
                file.readinto(m)
                mx[k, i, j] = m.x
                my[k, i, j] = m.y
                mz[k, i, j] = m.z

    file.close()
    return gi, gps, mx, my, mz

def GetFrameFromBinary(frames: int, gi: grid_info, raw_data: bytes, i: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if i < 0: i = 0
    elif i >= frames: i = frames - 1
    lat_s = gi.rows * gi.cols * gi.depth * VEC_SIZE
    raw_vecs = array.array("d")
    raw_vecs.frombytes(raw_data[i * lat_s: (i + 1) * lat_s])
    M = np.array(raw_vecs)
    mx, my, mz = M[0::3], M[1::3], M[2::3]
    return mx, my, mz

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

def FixCluster(input_dir, output_dir, dx, dy, dz, cut) -> bool:
    import ctypes
    _structure = ctypes.CDLL("/home/jose/.local/lib/atomistic3d/libatomistic3d.so")
    _structure.organize_clusters.argtypes = (ctypes.c_char_p, ctypes.c_char_p, ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double)
    _structure.organize_clusters.restype = ctypes.c_bool
    input_dir = input_dir + '\0'
    output_dir = output_dir + '\0'
    input_dir = ctypes.create_string_buffer(input_dir.encode("ascii"))
    output_dir = ctypes.create_string_buffer(output_dir.encode("ascii"))
    ret = _structure.organize_clusters(input_dir, output_dir, ctypes.c_double(dx), ctypes.c_double(dy), ctypes.c_double(dz), ctypes.c_double(cut))
    if not ret:
        raise FileNotFoundError
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
    angle[angle < 0] += 2
    #angle += np.pi
    #angle *= 180 / np.pi
    angle = (angle + 0) / 2.0
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

def GetClosest(arr: list[float], needle: float) -> tuple[int, float]:
    closest = 0
    closest_i = 0
    for i, val in enumerate(arr):
        delta = (val - needle) ** 2
        delta_closest = (val - closest) ** 2
        if delta < delta_closest:
            closest_i = i
            closest = val
    return closest_i, closest

def CreateAnimation(output:str, cmd: CMDArgs, frames: int, gi: grid_info, gp: list[grid_site_params], raw: bytes):
    import matplotlib.animation as animation
    if cmd.USE_LATEX:
        FixPlot(cmd.WIDTH, cmd.HEIGHT)
    else:
        FixPlot_(cmd.WIDTH, cmd.HEIGHT)

    fig, ax = plt.subplots()
    mx, my, mz = GetFrameFromBinary(frames, gi, raw, 3)

    img = None
    max_x, min_x = (gi.cols - 1) * gp[0].lattice + gp[0].lattice / 2.0, -gp[0].lattice / 2.0
    max_y, min_y = (gi.rows - 1) * gp[0].lattice + gp[0].lattice / 2.0, -gp[0].lattice / 2.0
    if cmd.HSL:
        r, g, b = GetHSL(mx, my, mz * cmd.INVERT)
        rgb = np.zeros((gi.rows, gi.cols, 3))
        rgb[:, :, 0] = r.reshape((gi.rows, gi.cols))
        rgb[:, :, 1] = g.reshape((gi.rows, gi.cols))
        rgb[:, :, 2] = b.reshape((gi.rows, gi.cols))
        img = ax.imshow(rgb, origin="lower", extent=[min_x, max_x, min_y, max_y], interpolation=cmd.INTERPOLATION)
    else:
        img = ax.imshow(mz.reshape((gi.rows, gi.cols)), origin="lower", extent=[min_x, max_x, min_y, max_y], interpolation=cmd.INTERPOLATION, vmin=-1, vmax=1, cmap=cmap)

    if cmd.PLOT_DEF:
        from matplotlib.collections import PatchCollection
        from matplotlib.patches import Rectangle
        colors = ["#00000000", "#DB850050", "#00610050", "#0000DB50", "#DBDB0050", "#00000050"]
        defects = ClusterDefects(gp)
        for y in range(gi.rows):
            for x in range(gi.cols):
                l = gp[y * gi.cols + x].lattice
                index = defects.index(gp[y * gi.cols + x])
                rect = plt.Rectangle((x * l - l / 2.0, y * l - l / 2.0), l, l,
                                     facecolor=colors[index % len(colors)])
                ax.add_patch(rect)

    vecs = None
    if cmd.PLOT_ARROWS:
        x, y, x_in, y_in, facx, facy = GetPosition(gi.rows, gi.cols, cmd.REDUCE_FACTOR, gp[0].lattice)
        mx_, my_ = GetVecsFromXY(mx, my, x_in, y_in)
        vecs = ax.quiver(x, y, mx_, my_, pivot="mid", units="xy", scale= 1.0 / np.sqrt(facx ** 2.0 + facy ** 2.0))

    ax.set_xlabel("x(nm)")
    ax.set_ylabel("y(nm)")
    ax.set_xticklabels((f"{i / 1.0e-9:.1f}" for i in ax.get_xticks()))
    ax.set_yticklabels((f"{i / 1.0e-9:.1f}" for i in ax.get_yticks()))
    def use_for_animate(i):
        print(f"{i / frames * 100.0:.2f}%")
        mx, my, mz = GetFrameFromBinary(frames, gi, raw, i)

        if cmd.PLOT_ARROWS:
            x, y, x_in, y_in, facx, facy = GetPosition(gi.rows, gi.cols, cmd.REDUCE_FACTOR, gp[0].lattice)
            mx_, my_ = GetVecsFromXY(mx, my, x_in, y_in)
            vecs.set_UVC(mx_, my_)
        if cmd.HSL:
            r, g, b = GetHSL(mx, my, mz * cmd.INVERT)
            rgb = np.zeros((gi.rows, gi.cols, 3))
            rgb[:, :, 0] = r.reshape((gi.rows, gi.cols))
            rgb[:, :, 1] = g.reshape((gi.rows, gi.cols))
            rgb[:, :, 2] = b.reshape((gi.rows, gi.cols))
            img.set_array(rgb)
        else:
            img.set_array(mz.reshape((gi.rows, gi.cols)))

    ani = animation.FuncAnimation(fig, use_for_animate, frames=frames)
    ani.save(output, fps=cmd.FPS, dpi=cmd.DPI)

def CreateAnimationFromFrames(base_dir: str, output: str, cmd: CMDArgs, gi: grid_info, gp: list[grid_site_params]):
    import matplotlib.animation as animation
    import os, cv2

    images_path = [i.name for i in os.scandir(base_dir) if i.name.endswith(".jpg")]
    images_path.sort(key=lambda x: float(x[x.find("_") + 1: x.find(".jpg")]))
    frames = [cv2.imread(f"{base_dir}/{i}") for i in images_path]

    fig, ax = plt.subplots()
    lattice = gp[0].lattice
    rows = len(frames[0])
    cols = len(frames[0][0])
    min_x, max_x = -lattice / 2, (cols - 1) * lattice + lattice / 2
    min_y, max_y = -lattice / 2, (rows - 1) * lattice + lattice / 2
    img = ax.imshow(frames[0], extent=[min_x, max_x, min_y, max_y])
    ax.set_xticklabels((f"{i/1e-9:.0f}" for i in ax.get_xticks()))
    ax.set_yticklabels((f"{i/1e-9:.0f}" for i in ax.get_yticks()))
    ax.set_xlabel("$x$(nm)")
    ax.set_ylabel("$y$(nm)")

    def animate(i):
        img.set_array(frames[i])
        print(i)
    ani = animation.FuncAnimation(fig, animate, frames=len(frames))
    ani.save(output, fps=cmd.FPS)

def ClusterDefects(gp: list[grid_site_params]) -> list[grid_site_params]:
    class defecttype:
        def __init__(self):
            self.counter = 0
            self.gp = None
        def __eq__(self, other):
            return self.gp == other.gp
        def __str__(self):
            return f"counter: {self.counter}"

    data = []

    for g in gp:
        df = defecttype()
        df.gp = g
        if df in data:
            data[data.index(df)].counter += 1
        else:
            df.counter = 1
            data.append(df)

    data.sort(key= lambda x: x.counter, reverse=True)
    for i in data:
        print(i)

    sites = [[] for _ in data]
    for g in gp:
        df = defecttype()
        df.gp = g
        i = data.index(df)
        if len(sites[i]) > 0:
            continue
        sites[i].append(g)

    return [sites[i][0] for i in range(len(sites))]

def GetClusterData(bdir: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    frames, gi, gp, raw = ReadAnimationBinary(f"{bdir}integrate_evolution.dat")
    lattice = gi.lattice

    try:
        data = pd.read_csv(f"{bdir}clusters_org.dat", header=None)
    except FileNotFoundError:
        FixCluster(f"./{bdir}clusters.dat", f"./{bdir}clusters_org.dat", gi.cols * lattice, gi.rows * lattice, gi.depth * lattice, 1e8)
        data = pd.read_csv(f"{bdir}clusters_org.dat", header=None)

    skip = 4 if (len(data.keys()) - 1) % 4 == 0 else 3
    start = 5 if (len(data.keys()) - 1) % 4 == 0 else 4
    xs = data.iloc[:, start::skip].to_numpy()
    ys = data.iloc[:, (start + 1)::skip].to_numpy()
    zs = data.iloc[:, (start + 2)::skip].to_numpy()
    if (len(data.keys()) - 1) % 4 == 0:
        ss = data.iloc[:, (start + 3)::skip].to_numpy()
    else:
        ss = np.ones_like(xs)
    min_x, max_x = 0, gi.cols * lattice
    min_y, max_y = 0, gi.rows * lattice

    dt = data.iloc[1, 0] - data.iloc[0, 0]
    vx = np.diff(xs, axis=0)
    vy = np.diff(ys, axis=0)
    vx = np.concatenate([vx, [vx[-1, :]]])
    vy = np.concatenate([vy, [vy[-1, :]]])

    bd_t, bd_p = np.where(vy > max_y / 2)
    bu_t, bu_p = np.where(vy < -max_y / 2)

    bl_t, bl_p = np.where(vx > max_x / 2)
    br_t, br_p = np.where(vx < -max_x / 2)

    factor = np.ones_like(vx[br_t, br_p]) * 2.0
    before = br_t - 1
    after = br_t + 1
    factor[before < 0] = 1
    factor[after >= xs.shape[1]] = 1
    before[before < 0] = 0
    after[after >= xs.shape[0]] = xs.shape[0] - 1
    vx[br_t, br_p] = (xs[after, br_p] + max_x - xs[before, br_p]) / factor

    factor = np.ones_like(vx[bl_t, bl_p]) * 2.0
    before = bl_t - 1
    after = bl_t + 1
    factor[before < 0] = 1
    factor[after >= xs.shape[1]] = 1
    before[before < 0] = 0
    after[after >= xs.shape[0]] = xs.shape[0] - 1
    vx[bl_t, bl_p] = (xs[after, bl_p] - max_x - xs[before, bl_p]) / factor

    factor = np.ones_like(vy[bu_t, bu_p]) * 2.0
    before = bu_t - 1
    after = bu_t + 1
    factor[before < 0] = 1
    factor[after >= ys.shape[1]] = 1
    before[before < 0] = 0
    after[after >= ys.shape[0]] = ys.shape[0] - 1
    vy[bu_t, bu_p] = (ys[after, bu_p] + max_y - ys[before, bu_p]) / factor

    factor = np.ones_like(vy[bd_t, bd_p]) * 2.0
    before = bd_t - 1
    after = bd_t + 1
    factor[before < 0] = 1
    factor[after >= ys.shape[1]] = 1
    before[before < 0] = 0
    after[after >= ys.shape[0]] = ys.shape[0] - 1
    vy[bd_t, bd_p] = (ys[after, bd_p] - max_y - ys[before, bd_p]) / factor

    #vx[bl_t, bl_p] = 0#-((xs[bl_t - 1, bl_p] + max_x) - xs[bl_t, bl_p]) / 1
    #vy[bu_t, bu_p] = 0#-((ys[bu_t - 1, bu_p] - max_y) - ys[bu_t, bu_p]) / 1
    #vy[bd_t, bd_p] = 0#-((ys[bd_t - 1, bd_p] + max_y) - ys[bd_t, bd_p]) / 1
    return data[0].to_numpy(), xs, ys, zs, vx / dt, vy / dt, ss

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
                     #"axes.autolimit_mode": "round_numbers",
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
                     #"axes.autolimit_mode": "round_numbers",
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
                     "mathtext.fontset": "custom"}) #"cm"
