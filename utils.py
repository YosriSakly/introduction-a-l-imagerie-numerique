import numpy as np
import cv2
from PIL import Image
import os, os.path

import images_fusion


def load_images(path, gray=True, align=False):
    imgs = []
    # Load images from the path with different focus
    valid_images = [".jpg", ".png", ".gif"]
    for f in os.listdir(path):
        ext = os.path.splitext(f)[1]
        if ext.lower() not in valid_images:
            continue
        if ext.lower() == ".gif":
            gif = Image.open((os.path.join(path, f)))
            imgs.append(np.array(gif.convert('RGB')))
        else:
            imgs.append(cv2.imread(os.path.join(path, f)))
    if align:
        alignMTB = cv2.createAlignMTB()
        alignMTB.process(imgs, imgs)
    images = tuple(imgs)
    if gray:
        images = [cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) for img in images]
    return images


def fuse_images(images, avg_size=31, lapsize=1, rg=5, sigmag=5, eps1=0.1, r1=7, eps2=1, r2=7, plot=False):
    if plot:
        for i in range(len(images)):
            cv2.imshow("image " + str(i + 1), images[i])
    # Base / details decomposition
    bases_tuple, details_tuple = images_fusion.basis_details_decomposition(images, avg_size)
    if plot:
        for i in range(len(images)):
            cv2.imshow('Base Layer B' + str(i + 1), cv2.convertScaleAbs(bases_tuple[i]))
            # cv2.waitKey(0); cv2.destroyAllWindows(); cv2.waitKey(1)
            cv2.imshow('Details Layer B' + str(i + 1), cv2.convertScaleAbs(details_tuple[i]))
            # cv2.waitKey(0); cv2.destroyAllWindows(); cv2.waitKey(1)
    # Compute the Laplacians
    saliency_tuple = images_fusion.saliency_maps(images, lapsize, rg, sigmag)
    if plot:
        for i in range(len(images)):
            cv2.imshow("saliency" + str(i + 1), saliency_tuple[i])
            # cv2.waitKey(0);
            # cv2.destroyAllWindows();
            # cv2.waitKey(1)
    # Get the weight maps
    weight_maps = images_fusion.weight_map_construction(saliency_tuple)
    if plot:
        for i in range(len(images)):
            cv2.imshow("weight maps " + str(i + 1), weight_maps[i] * 255)
            # cv2.waitKey(0); cv2.destroyAllWindows(); cv2.waitKey(1)
    # Refined weight maps using guided filtering
    refined_wm_basis = images_fusion.refined_weight_maps(weight_maps, images, r1, eps1)
    refined_wm_details = images_fusion.refined_weight_maps(weight_maps, images, r2, eps2)
    if plot:
        for i in range(len(images)):
            cv2.imshow("Refined weight map (Basis) " + str(i + 1), refined_wm_basis[i] * 255)
        for i in range(len(images)):
            cv2.imshow("Refined weight map (Details) " + str(i + 1), refined_wm_details[i] * 255)
    # Normalization of weight maps
    refined_normalized_bases = images_fusion.weight_maps_normalization(refined_wm_basis)
    refined_normalized_details = images_fusion.weight_maps_normalization(refined_wm_details)
    # Fused basis and fused details
    fused_bases = images_fusion.images_weighted_average(refined_normalized_bases, bases_tuple)
    fused_details = images_fusion.images_weighted_average(refined_normalized_details, details_tuple)
    fused_image = fused_bases + fused_details
    fused_image_uint8 = cv2.convertScaleAbs(fused_image)
    if plot:
        cv2.imshow("Fused image", fused_image_uint8)
    return fused_image_uint8
