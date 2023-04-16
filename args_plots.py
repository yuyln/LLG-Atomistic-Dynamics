import argparse
from struct import *
import array
import mmap
import pprint
import numpy as np


class CMDArgs:
    def __init__(self, inp, out):
        parser = argparse.ArgumentParser(description="Configure plots parameters via CLI")
        parser.add_argument("-input", default=inp, nargs="?", type=str)
        parser.add_argument("-output", default=out, nargs="?", type=str)
        parser.add_argument("-DPI", default=250, nargs="?", type=int)
        parser.add_argument("-arrows", default=False, nargs="?", type=bool)
        parser.add_argument("-factor", default=1, nargs="?", type=int)
        parser.add_argument("-fps", default=60, nargs="?", type=int)
        parser.add_argument("-anisotropy", default=False, nargs="?", type=bool)
        parser.add_argument("-pinning", default=False, nargs="?", type=bool)
        parser.add_argument("-anisotropy-color", default="black", nargs="?", type=str)
        parser.add_argument("-pinning-color", default="yellow", nargs="?", type=str)
        parser.add_argument("-interpolation", default="nearest", nargs="?", type=str)
        parser.add_argument("-latex", default=False, nargs="?", type=bool)
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

    def print(self):
        pprint.pprint(self.args)


def ReadAnimationBinary(path: str) -> list[int, int, int, int, float, float, np.array]:
    file = open(path, "r")
    raw_data = mmap.mmap(file.fileno(), 0, access=mmap.ACCESS_READ)
    rows, cols, frames, cut = unpack("iiii", raw_data[:16])
    dt, lattice = unpack("dd", raw_data[16:32])
    print(rows, cols, frames, cut, dt, lattice)
    file.close()
    return None

test = CMDArgs("./output/integration_fly.bin", "./imgs/out.png")
test.print()
ReadAnimationBinary(test.INPUT_FILE)
