import cv2
import numpy as np
import matplotlib.pyplot as plt
import os, os.path
import scipy.ndimage
from PIL import Image
import importlib

import utils
import images_fusion



##################### Loadings and other prerequisites #############################################################

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
eps1 = 0.1
r1 = 7
eps2 = 1
r2 =7

fused = utils.fuse_images(images, avg_size, lapsize, rg, sigmag, eps1, r1, eps2, r2, plot=True)




