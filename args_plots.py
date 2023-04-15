import argparse
import pprint

parser = argparse.ArgumentParser(description="Configure plots parameters via CLI")
parser.add_argument("-input", default="./output/out.bin", nargs=1)
parser.add_argument("-output", default="./imgs/out_lattice.png", nargs=1)
parser.add_argument("-DPI", default=250, nargs=1)
parser.add_argument("-arrows", default=False, nargs=1)
parser.add_argument("-factor", default=1, nargs=1)
parser.add_argument("-fps", default=60, nargs=1)
args = parser.parse_args()
pprint.pprint(vars(args))
