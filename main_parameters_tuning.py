import cv2
import numpy as np
import os, os.path
import scipy.ndimage
from scipy import misc
from PIL import Image
import importlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
# Plot parameters
plt.rcParams.update({"font.size": 22})
plt.rcParams.update({"lines.linewidth": 5})
plt.rcParams.update({"lines.markersize": 10})

import utils
import images_fusion
import metrics

importlib.reload(metrics)



# Path to the data
path = os.getcwd() + "/Data/rose"

# Load images
images = utils.load_images(path, gray=True, align=False)

# Size of kernel for basis / details decomposition
avg_size=31
# Laplacian filter size
lapsize = 1
# Parameters for Gaussian lowpass filter
rg = 5
sigmag = 5
# Refined weight maps using guided filtering parameters
eps1 = 0.3
r1 = 3
eps2 = 10e-6
r2 = 3

fused = utils.fuse_images(images, avg_size, lapsize, rg, sigmag, eps1, r1, eps2, r2, plot=True)
misc.imsave(os.getcwd() + "/Rose_Fused_Params.png", fused)

# ############################## 3D PARAMETER SELECTION EXAMPLE #######################################################
r1_grid = np.arange(1, 50)
r2_grid = np.arange(1, 50)
R1, R2 = np.meshgrid(r1_grid, r2_grid)
Z = np.zeros((r1_grid.shape[0], r2_grid.shape[0]))
count = 0
for i in range(r1_grid.shape[0]):
    for j in range(r2_grid.shape[0]):
        fused = utils.fuse_images(images, avg_size, lapsize, rg, sigmag, eps1, i, eps2, j, plot=False)
        # Z[i, j] = metrics.Q_mutual_information(images[0], images[1], fused)
        Z[i, j] = metrics.Q_ssim(images[0], images[1], fused)
        count += 1
        print(r1_grid.shape[0] * r2_grid.shape[0] - count)




# Plot 3D results
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(R1, R2, Z, cmap=cm.coolwarm, linewidth=0, antialiased=False)
ax.set_xlabel("$r_1$", labelpad=20)
ax.set_ylabel("$r_2$", labelpad=20)
ax.set_zlabel("$Q_{SSIM}$", labelpad=10)
# plt.title("Selection of $r_1$ and $r_2$ with $\epsilon_1 = 0.3$ and $\epsilon_2 = 1")
plt.title("Selection of $r_1$ and $r_2$ with $\epsilon_1 = 0.3$ and $\epsilon_2 = 10^{-6}$")



#
#
# # Compare values of r1
# r1_grid = np.arange(1, 70, 1)
# qimr1_list = []
# qssimr1_list = []
# for r in r1_grid:
#     fused = utils.fuse_images(images, avg_size, lapsize, rg, sigmag, eps1, r, eps2, r2, plot=False)
#     qimr1_list.append(metrics.Q_mutual_information(images[0], images[1], fused))
#     qssimr1_list.append(metrics.Q_ssim(images[0], images[1], fused, kernel_size=40))
#     print(r)
#
# norm_qimr1 = np.array(qimr1_list) / (np.array(qimr1_list).max())
# norm_qssimr1 = np.array(qssimr1_list) / (np.array(qssimr1_list).max())
# plt.figure()
# plt.plot(r1_grid, norm_qimr1, marker="o", label="$Q_{IM}$")
# plt.plot(r1_grid, norm_qssimr1, marker="o", label="$Q_{SSIM}$")
# plt.xlabel("$r_1$")
# plt.ylabel("Normalized metric")
# plt.legend()
#
# r1 = 25
#
#
# # Compare values of r2
# r2_grid = np.arange(1, 70, 1)
# qimr2_list = []
# qssimr2_list = []
# for r in r1_grid:
#     fused = utils.fuse_images(images, avg_size, lapsize, rg, sigmag, eps1, r1, eps2, r, plot=False)
#     qimr2_list.append(metrics.Q_mutual_information(images[0], images[1], fused))
#     qssimr2_list.append(metrics.Q_ssim(images[0], images[1], fused))
#     print(r)
#
# norm_qimr2 = np.array(qimr2_list) / (np.array(qimr2_list).max())
# norm_qssimr2 = np.array(qssimr2_list) / (np.array(qssimr2_list).max())
# plt.figure()
# plt.plot(r2_grid, norm_qimr2, marker="o", label="$Q_{IM}$")
# plt.plot(r2_grid, norm_qssimr2, marker="o", label="$Q_{SSIM}$")
# plt.xlabel("$r_2$")
# plt.ylabel("Normalized metric")
# plt.legend()
# plt.legend()
#
# r2 = 7
# # Compare values of eps1
# eps1_grid = np.geomspace(start = 1e-7, stop=1, num=70)
# qimeps1_list = []
# qssimeps1_list = []
# count = 0
# for eps in eps1_grid:
#     fused = utils.fuse_images(images, avg_size, lapsize, rg, sigmag, eps, r1, eps2, r2, plot=False)
#     qimeps1_list.append(metrics.Q_mutual_information(images[0], images[1], fused))
#     qssimeps1_list.append(metrics.Q_ssim(images[0], images[1], fused))
#     print(count)
#     count += 1
#
# norm_qimeps1 = np.array(qimeps1_list) / (np.array(qimeps1_list).max())
# norm_qssimeps1 = np.array(qssimeps1_list) / (np.array(qssimeps1_list).max())
# plt.figure()
# plt.plot(eps1_grid, norm_qimeps1, marker="o", label="$Q_{IM}$")
# plt.plot(eps1_grid, norm_qssimeps1, marker="o", label="$Q_{SSIM}$")
# plt.xlabel("$\epsilon_1$")
# plt.ylabel("Normalized metric")
# plt.xscale("log")
# plt.legend()
# plt.legend()
#
#
# eps1 = 0.1
#
#
#
# # Compare values of eps2
# eps2_grid = np.geomspace(start = 1e-7, stop=1, num=70)
# qimeps2_list = []
# qssimeps2_list = []
# count = 0
# for eps in eps2_grid:
#     fused = utils.fuse_images(images, avg_size, lapsize, rg, sigmag, eps1, r1, eps, r2, plot=False)
#     qimeps2_list.append(metrics.Q_mutual_information(images[0], images[1], fused))
#     qssimeps2_list.append(metrics.Q_ssim(images[0], images[1], fused))
#     print(count)
#     count += 1
#
# norm_qimeps2 = np.array(qimeps2_list) / (np.array(qimeps2_list).max())
# norm_qssimeps2 = np.array(qssimeps2_list) / (np.array(qssimeps2_list).max())
# plt.figure()
# plt.plot(eps2_grid, norm_qimeps2, marker="o", label="$Q_{IM}$")
# plt.plot(eps2_grid, norm_qssimeps2, marker="o", label="$Q_{SSIM}$")
# plt.xlabel("$\epsilon_2$")
# plt.ylabel("Normalized metric")
# plt.xscale("log")
# plt.legend()
# plt.legend()
#
#
# fig, axes = plt.subplots(nrows=2, ncols=2)
# axes[0, 0].plot(r1_grid, norm_qimr1, marker="o", label="$Q_{IM}$")
# axes[0, 0].plot(r1_grid, norm_qssimr1, marker="o", label="$Q_{SSIM}$")
# axes[0, 0].set_xlabel("$r_1$")
# axes[0, 0].set_ylabel("Normalized metric")
# axes[0, 0].legend()
# axes[1, 0].plot(r2_grid, norm_qimr2, marker="o", label="$Q_{IM}$")
# axes[1, 0].plot(r2_grid, norm_qssimr2, marker="o", label="$Q_{SSIM}$")
# axes[1, 0].set_xlabel("$r_2$")
# axes[1, 0].set_ylabel("Normalized metric")
# axes[1, 0].legend()
# axes[0, 1].plot(eps1_grid, norm_qimeps1, marker="o", label="$Q_{IM}$")
# axes[0, 1].plot(eps1_grid, norm_qssimeps1, marker="o", label="$Q_{SSIM}$")
# axes[0, 1].set_xlabel("$\epsilon_1$")
# axes[0, 1].set_ylabel("Normalized metric")
# axes[0, 1].set_xscale("log")
# axes[0, 1].legend()
# axes[1, 1].plot(eps2_grid, norm_qimeps2, marker="o", label="$Q_{IM}$")
# axes[1, 1].plot(eps2_grid, norm_qssimeps2, marker="o", label="$Q_{SSIM}$")
# axes[1, 1].set_xlabel("$\epsilon_2$")
# axes[1, 1].set_ylabel("Normalized metric")
# axes[1, 1].set_xscale("log")
# axes[1, 1].legend()