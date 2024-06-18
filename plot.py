import utils
import matplotlib.pyplot as plt
import numpy as np

cmd = utils.CMDArgs("dummy", "dummy")
frames, gi, gp, raw = utils.ReadAnimationBinary("/home/jose/dados/monroe/vary_dj/data/0.27000/4.40000e+10/integrate_evolution.dat")
print(utils.ClusterDefects(gp))
utils.CreateAnimation("dummy.mp4", cmd, frames, gi, gp, raw)
