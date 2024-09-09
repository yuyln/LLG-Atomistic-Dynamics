import utils
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import pandas as pd

from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle

def fix_cluster(input_dir, output_dir, dx, dy, cut, size):
    import ctypes
    _structure = ctypes.CDLL("/home/jose/.local/lib/atomistic/libatomistic.so")
    _structure.organize_clusters.argtypes = (ctypes.c_char_p, ctypes.c_char_p, ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_bool)
    input_dir = input_dir + '\0'
    output_dir = output_dir + '\0'
    input_dir = ctypes.create_string_buffer(input_dir.encode("ascii"))
    output_dir = ctypes.create_string_buffer(output_dir.encode("ascii"))
    _structure.organize_clusters(input_dir, output_dir, ctypes.c_double(dx), ctypes.c_double(dy), ctypes.c_double(cut), ctypes.c_bool(size))

utils.FixPlot(8, 8)
cmd = utils.CMDArgs("dummy", "dummy")
frames, gi, gp, raw = utils.ReadAnimationBinary("integrate_evolution.dat")
utils.CreateAnimation("./test.mp4", cmd, frames, gi, gp, raw)
