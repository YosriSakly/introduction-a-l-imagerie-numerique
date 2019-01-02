import cv2
import numpy as np
import matplotlib.pyplot as plt
import os, os.path
import scipy.ndimage
from PIL import Image


import images_fusion




##################### Loadings and other prerequisites #############################################################

# Path to the data
imgs = []
path = os.getcwd() + "/Data/set4"

# Load images from the path with different focus

valid_images = [".jpg", ".png", ".gif"]
for f in os.listdir(path):
    ext = os.path.splitext(f)[1]
    if ext.lower() not in valid_images:
        continue
    if ext.lower() == ".gif":
        ## Read the gif from disk to `RGB`s using `imageio.miread` 

        gif = Image.open((os.path.join(path, f)))
        imgs.append(np.array(gif.convert('RGB')))
    else:
        imgs.append(cv2.imread(os.path.join(path, f)))
images = tuple(imgs)
# Plot them
for i in range(len(images)):
    cv2.imshow("image " + str(i + 1), images[i])
    # cv2.waitKey(0);
    # cv2.destroyAllWindows();
    # cv2.waitKey(1)



alignMTB = cv2.createAlignMTB()
alignMTB.process(imgs, imgs)

# Convert to grayscale
images = [cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) for img in images]

# ################### A : Two scales image decomposition ###########################################################
#images_gray=tuple([cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) for img in images])
#images=tuple([img.astype(float) for img in images])
#images_gray=tuple([img.astype(float) for img in images_gray])

# Size of averaging kernel for basis/details decomposition
avg_size = 31
# Get basis and details images
bases_tuple, details_tuple = images_fusion.basis_details_decomposition(images, avg_size)
# See result of basis/details decomposition
for i in range(len(images)):
    cv2.imshow('Base Layer B' + str(i+1) , cv2.convertScaleAbs(bases_tuple[i]))
    # cv2.waitKey(0); cv2.destroyAllWindows(); cv2.waitKey(1)
    cv2.imshow('Details Layer B' + str(i+1) , cv2.convertScaleAbs(details_tuple[i]))
    # cv2.waitKey(0); cv2.destroyAllWindows(); cv2.waitKey(1)


# ################### B : Weight map construction with guided filtering ############################################

# Laplacian filter size
lapsize = 1
# Parameters for Gaussian lowpass filter
rg = 5
sigmag = 5
# Compute laplacians
saliency_tuple = images_fusion.saliency_maps(images, lapsize, rg, sigmag)
for i in range(len(images)):
    cv2.imshow("saliency" + str(i + 1), saliency_tuple[i])
    # cv2.waitKey(0);
    # cv2.destroyAllWindows();
    # cv2.waitKey(1)


# Get the weight maps
weight_maps = images_fusion.weight_map_construction(saliency_tuple)
# Vizualize weight map, we multiply by 255 to distinguish the pixel equal to 1 from those equal to 0
for i in range(len(images)):
    cv2.imshow("weight maps "+ str(i+1), weight_maps[i]*255)
    # cv2.waitKey(0); cv2.destroyAllWindows(); cv2.waitKey(1)



# Refined weight maps using guided filtering
eps1 = 0.1
r1 = 7
refined_wm_basis = images_fusion.refined_weight_maps(weight_maps, images, r1, eps1)
eps2 = 1
r2 =7
refined_wm_details = images_fusion.refined_weight_maps(weight_maps, images, r2, eps2)



# Normalization of weight maps
refined_normalized_bases = images_fusion.weight_maps_normalization(refined_wm_basis)
refined_normalized_details = images_fusion.weight_maps_normalization(refined_wm_details)



# Fused basis and fused details
fused_bases = images_fusion.images_weighted_average(refined_normalized_bases, bases_tuple)
fused_details = images_fusion.images_weighted_average(refined_normalized_details, details_tuple)



# Fused image
fused_image = fused_bases + fused_details
fused_image_uint8 = cv2.convertScaleAbs(fused_image)
cv2.imshow("Fused image", fused_image_uint8)
# cv2.waitKey(0); cv2.destroyAllWindows(); cv2.waitKey(1)






