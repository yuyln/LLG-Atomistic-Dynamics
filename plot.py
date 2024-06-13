import utils
import matplotlib.pyplot as plt
import numpy as np

cmd = utils.CMDArgs("dummy", "dummy")
frames, gi, gp, raw = utils.ReadAnimationBinary("integrate_evolution.dat")
#print(utils.ClusterDefects(gp))
utils.CreateAnimation("dummy.mp4", cmd, frames, gi, gp, raw)
