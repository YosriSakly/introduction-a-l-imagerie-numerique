import cv2
import numpy as np
import matplotlib.pyplot as plt
import os, os.path
import scipy.ndimage as ndimage
from PIL import Image
import importlib


import pyramid_fusion
importlib.reload(pyramid_fusion)




##################### Loadings and other prerequisites #############################################################

# # Path to the data
# imgs = []
# path = os.getcwd() + "/Data/set4"
#
# # Load images from the path with different focus
#
# valid_images = [".jpg", ".png", ".gif"]
# for f in os.listdir(path):
#     ext = os.path.splitext(f)[1]
#     if ext.lower() not in valid_images:
#         continue
#     if ext.lower() == ".gif":
#         ## Read the gif from disk to `RGB`s using `imageio.miread`
#
#         gif = Image.open((os.path.join(path, f)))
#         imgs.append(np.array(gif.convert('RGB')))
#     else:
#         imgs.append(cv2.imread(os.path.join(path, f)))
# images = tuple(imgs)
# # Plot them
# for i in range(len(images)):
#     cv2.imshow("image " + str(i + 1), images[i])
#     # cv2.waitKey(0);
#     # cv2.destroyAllWindows();
#     # cv2.waitKey(1)



##################### Loadings and other prerequisites #############################################################

# Path to the data
imgs = []
path = os.getcwd() + "/Data/rose"

# Load images from the path with different focus

valid_images = [".jpg", ".png"]
for f in os.listdir(path):
    ext = os.path.splitext(f)[1]
    if ext.lower() not in valid_images:
        continue
    imgs.append(cv2.imread(os.path.join(path, f)))
images = tuple(imgs)
# Plot them
for i in range(len(images)):
    cv2.imshow("image " + str(i + 1), images[i])
    # cv2.waitKey(0);
    # cv2.destroyAllWindows();
    # cv2.waitKey(1)


# alignMTB = cv2.createAlignMTB()
# alignMTB.process(imgs, imgs)

# Convert to grayscale
imgs_gray = [cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) for img in images]

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
fused_equalized = cv2.equalizeHist(fused_image)
cv2.imshow("Fused image", fused_equalized)