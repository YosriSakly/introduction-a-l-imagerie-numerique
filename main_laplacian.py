import cv2
import numpy as np
import matplotlib.pyplot as plt
import os, os.path
import scipy.ndimage as ndimage
from PIL import Image
import importlib
import utils


import pyramid_fusion
importlib.reload(pyramid_fusion)




##################### Loadings and other prerequisites #############################################################

# Path to the data
path = os.getcwd() + "/Data/rose"

# Load images
imgs_gray = utils.load_images(path, gray=True, align=False)


# Number of levels
levels = 4



# Set size of images to avoid problems in Laplacian pyramids computations
imgs_gray = pyramid_fusion.adapt_number_of_pixels(imgs_gray, levels)

# Create Laplacian pyramids
gaussian_pyramids = pyramid_fusion.build_gaussian_pyramids(imgs_gray, levels)
laplacian_pyramids = pyramid_fusion.gaussian_to_laplacian_pyramids(gaussian_pyramids)

# Reconstruct test
reconstructed = pyramid_fusion.reconstruct(laplacian_pyramids[0])
cv2.imshow("Reconstruction test", reconstructed)
cv2.imshow("Original", imgs_gray[0])

# Fuse pyramids
fused_pyramids = pyramid_fusion.fuse_pyramids(laplacian_pyramids, kernel_size=5, kernel_coef=30)

# Reconstruct
fused_image = pyramid_fusion.reconstruct(fused_pyramids)
# fused_equalized = cv2.equalizeHist(fused_image)
cv2.imshow("Fused image", fused_image)

