import utils
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans, DBSCAN, OPTICS, AgglomerativeClustering, AffinityPropagation

cmd = utils.CMDArgs("dummy", "dummy")
frames, gi, gp, raw = utils.ReadAnimationBinary("integrate_evolution.dat")
print(frames)
mx, my, mz = utils.GetFrameFromBinary(frames, gi, raw, frames - 1)
x, y, _, _, _, _ = utils.GetPosition(gi.rows, gi.cols, 1, 1)
X = np.zeros((gi.rows * gi.cols, 5))
X[:, 0] = x / gi.cols * 2 - 1
X[:, 1] = y / gi.rows * 2 - 1

X[:, 2] = mx
X[:, 3] = my
X[:, 4] = mz


fig, ax = plt.subplots(ncols=3)
ax[0].imshow(mz.reshape((gi.rows, gi.cols)), cmap=utils.cmap, extent=[-0.5, gi.cols - 0.5, -0.5, gi.rows - 0.5], origin="lower")
#ax[1].imshow(mz.reshape((gi.rows, gi.cols)), cmap=utils.cmap, extent=[-0.5, gi.cols - 0.5, -0.5, gi.rows - 0.5], origin="lower")


kmeans = KMeans(n_clusters=3).fit(X)
y = (kmeans.cluster_centers_[:, 1] + 1) * 0.5 * gi.rows
x = (kmeans.cluster_centers_[:, 0] + 1) * 0.5 * gi.cols
ax[0].scatter(x, y)

labels = kmeans.labels_

ax[1].imshow(labels.reshape((gi.rows, gi.cols)), extent=[-0.5, gi.cols - 0.5, -0.5, gi.rows - 0.5], origin="lower", cmap="Set1")

X[X[:, 4] > 0] = 1
X[X[:, 4] <= 0] = -1
dbscan = DBSCAN(eps=0.08).fit(X)
labels = dbscan.labels_
ax[2].imshow(labels.reshape((gi.rows, gi.cols)), extent=[-0.5, gi.cols - 0.5, -0.5, gi.rows - 0.5], origin="lower", cmap="Set1")
print(dbscan.components_.shape)

#affinity = AffinityPropagation().fit(X)
#labels = affinity.labels_
#ax[1][1].imshow(labels.reshape((gi.rows, gi.cols)), extent=[-0.5, gi.cols - 0.5, -0.5, gi.rows - 0.5], origin="lower", cmap="Set1")

plt.show()
