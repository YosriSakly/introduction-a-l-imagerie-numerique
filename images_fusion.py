import cv2
import numpy as np
import matplotlib.pyplot as plt
import os, os.path
import scipy.ndimage
from PIL import Image


# Function for basis_details_decomposition (Corresponds to equation 10 and 11 from the article)
def basis_details_decomposition(images_tuple, avg_size=31):
    kernel = np.ones((avg_size, avg_size), np.float32) / avg_size ** 2
    basis_list = []
    details_list = []
    for im in images_tuple:
        # basis=scipy.ndimage.correlate(im, kernel, mode='nearest')
        basis = np.array(cv2.blur(im, (avg_size, avg_size), borderType=cv2.BORDER_REPLICATE), np.float64)
        # basis=np.array(cv2.boxFilter(im,-1,(avg_size,avg_size) ), np.float64)
        details = np.array(im, np.float64) - basis
        basis_list.append(basis)
        details_list.append(im - basis)
    return tuple(basis_list), tuple(details_list)


# Function for saliencies (laplacian filter + gaussian lowpass filter)
# Corresponds to equation 13 in the article
def saliency_maps(images_tuple, lapsize, rg, sigmag):
    saliency_list = []
    for im in images_tuple:
        # laplacian = abs(cv2.Laplacian(im, cv2.CV_64F))
        laplacian = abs(cv2.Sobel(im, cv2.CV_64F, 0, 1, ksize=1))
        saliency = cv2.GaussianBlur(laplacian, (2 * rg + 1, 2 * rg + 1), sigmag)
        saliency_list.append(saliency)
        s = np.array(saliency_list) + 1e-12
        saliency_list_normalized = list((s / np.sum(s, axis=0)))
    return tuple(saliency_list_normalized)


# Weight map function (corresponds to equation (14) from the arcticle)
def weight_map_construction(saliency_tuple):
    # Intialize weight maps to zero
    dims = saliency_tuple[0].shape
    nims = len(saliency_tuple)

    weight_maps = [np.zeros(dims, np.uint8) for i in range(0, nims)]
    # Loop over color channels (we have one weight map per channel)
    argmax = np.argmax(saliency_tuple, axis=0)
    # Use that information to fill the weight maps
    for i in range(0, nims):
        # Get the indicator for each pixel and fill the corresponding arrays
        weight_maps[i] = (argmax == i).astype(np.uint8)
    return weight_maps


# Produce refined weights maps using guided filtering (Function for equations 15 and 16 of the article)
def refined_weight_maps(guides_tuple, images_tuple, r, eps):
    nims = len(images_tuple)
    outputs = []
    for i in range(0, nims):
        gf = cv2.ximgproc.createGuidedFilter(images_tuple[i], r, eps)
        filtered = gf.filter(guides_tuple[i])

        outputs.append(filtered)
    return tuple(outputs)


# Pixel by pixel normalization
def weight_maps_normalization(weights_tuple):
    weights_list = np.array(list(weights_tuple))
    weights_list = cv2.convertScaleAbs(weights_list * 255)
    weights_list = weights_list.astype(float) / 255
    weights_list = weights_list + 1e-12

    weights_list_normalized = list(weights_list / np.sum(weights_list, axis=0))
    return tuple(weights_list_normalized)


def images_weighted_average(weights_tuple, images_tuple):
    # size=np.array(weights_tuple).shape+(1,)
    return np.sum(np.array(weights_tuple) * np.array(images_tuple), axis=0)