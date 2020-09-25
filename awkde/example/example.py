# coding: utf-8

"""
A 2D example for the adaptive width KDE.

Sample x1 from a gaussian and x2 from x*exp(-x).
Make a plot with the prediction and save the model to a JSON dump.
"""

import numpy as np
import scipy.stats as scs
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LogNorm

from awkde import GaussianKDE


rndgen = np.random.RandomState(seed=3575)  # ESEL
# Gaussian
mean = 3.
sigma = .25
# a^2 * x * exp(-a * x)
a = 100.

n_samples = 1000
logE_sam = rndgen.normal(mean, sigma, size=n_samples)

# From pythia8: home.thep.lu.se/~torbjorn/doxygen/Basics_8h_source.html
u1, u2 = rndgen.uniform(size=(2, n_samples))
sigma_sam = -np.log(u1 * u2) / a

# Shape must be (n_points, n_features)
sample = np.vstack((logE_sam, sigma_sam)).T

# Create KDE and fit it. Save model in JSON format
print("Fitting model to {} sample points.".format(n_samples))
kde = GaussianKDE(glob_bw="silverman", alpha=0.5, diag_cov=True)
kde.fit(sample)

# Save and load the model
outf = "./example_KDE.json"
print("Saving model to {}".format(outf))
kde.to_json(outf)
print("Loading same model from {}".format(outf))
kde = GaussianKDE.from_json(outf)

# Evaluate at dense grid
minx, maxx = np.amin(sample[:, 0]), np.amax(sample[:, 0])
miny, maxy = np.amin(sample[:, 1]), np.amax(sample[:, 1])

x = np.linspace(minx, maxx, 100)
y = np.linspace(miny, maxy, 100)

XX, YY = np.meshgrid(x, y)
grid_pts = np.array(list(map(np.ravel, [XX, YY]))).T

zz = kde.predict(grid_pts)
ZZ = zz.reshape(XX.shape)


# Evaluate true PDFs at same grid
def fx(x):
    return scs.norm.pdf(x, mean, sigma)


def fy(y):
    return a**2 * y * np.exp(-a * y)


fZ = (fx(grid_pts[:, 0]) * fy(grid_pts[:, 1])).reshape(XX.shape)

# Sample new points from KDE model
kde_sam = kde.sample(n_samples=100000, random_state=rndgen)

print("Making example plot.")
# Big plot on the left (2D KDE + points ) and three right (1D margins + hist)
fig = plt.figure(figsize=(10, 6))
gs = gridspec.GridSpec(3, 3)
axl = fig.add_subplot(gs[:, :2])
axrt = fig.add_subplot(gs[0, 2])
axrc = fig.add_subplot(gs[1, 2])
axrb = fig.add_subplot(gs[2, 2])

# Main plot
axl.pcolormesh(XX, YY, ZZ, cmap="Blues", norm=LogNorm())
axl.scatter(logE_sam, sigma_sam, marker=".", color="#353132",
            edgecolor="", s=30)
axl.set_title("KDE log PDF + original sample")

# Top right: truth with scatter
axrt.pcolormesh(XX, YY, fZ, cmap="Blues", norm=LogNorm())
axrt.scatter(logE_sam, sigma_sam, marker=".", color="#353132", s=1)
axrt.set_title("True log PDF + KDE sample")

# 1D x1, x2 hists. Hist very fine, so we get the shape of the PDF and don't
# have to integrate the KDE PDF numerically.
axrc.hist(kde_sam[:, 0], bins=250, normed=True, color="#353132")
axrc.plot(x, fx(x), color="#1e90ff")
axrb.hist(kde_sam[:, 1], bins=250, normed=True, color="#353132")
axrb.plot(y, fy(y), color="#1e90ff")
axrc.set_title("True 1D PDF + KDE sample")
axrb.set_title("True 1D PDF + KDE sample")

for axi in (axl, axrt):
    axi.set_xlim(minx, maxx)
    axi.set_ylim(0, maxy)
    axi.set_xlabel("x1")
    axi.set_ylabel("x2")

axrc.set_xlim(minx, maxx)
axrb.set_xlim(miny, maxy)
axrc.set_xlabel("x1")
axrb.set_xlabel("x2")

fig.tight_layout()
fig.savefig("example.png", dpi=50)
# plt.show()
