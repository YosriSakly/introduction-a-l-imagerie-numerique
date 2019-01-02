import cv2
from skimage import morphology
from skimage.filters.rank import entropy
import numpy as np


def gaussian_kernel(l=5, sig=1.):
    """
    creates gaussian kernel with side length l and a sigma of sig
    """
    ax = np.arange(-l // 2 + 1., l // 2 + 1.)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx**2 + yy**2) / (2. * sig**2))
    return kernel / np.sum(kernel)


def adapt_npixels(img, levels):
    divider = 2 ** levels
    nx = img.shape[0]
    ny = img.shape[1]
    nx_round = (nx // divider) * divider
    ny_round = (ny // divider) * divider
    img_copy = img.copy()
    print(nx_round)
    print(ny_round)
    return img_copy[:nx_round, :ny_round]


def adapt_number_of_pixels(images_tuple, levels):
    divider = 2 ** levels
    rounded = []
    for img in images_tuple:
        nx = img.shape[0]
        ny = img.shape[1]
        nx_round = (nx // divider) * divider
        ny_round = (ny // divider) * divider
        rounded.append(img.copy()[:nx_round, :ny_round])
    return rounded


def build_gaussian_pyramids(images_tuple, levels):
    G1 = images_tuple[0].copy()
    G2 = images_tuple[1].copy()
    pyramids = [[G1], [G2]]
    for i in range(levels):
        G1 = cv2.pyrDown(G1)
        G2 = cv2.pyrDown(G2)
        pyramids[0].append(G1)
        pyramids[1].append(G2)
    return pyramids


def gaussian_to_laplacian_pyramids(gaussian_pyramids):
    levels = len(gaussian_pyramids[0]) - 1
    pyramids = [[gaussian_pyramids[0][-1]], [gaussian_pyramids[1][-1]]]
    for i in range(levels, 0, -1):
        GE1 = cv2.pyrUp(gaussian_pyramids[0][i])
        GE2 = cv2.pyrUp(gaussian_pyramids[1][i])
        L1 = cv2.subtract(gaussian_pyramids[0][i - 1], GE1)
        L2 = cv2.subtract(gaussian_pyramids[1][i - 1], GE2)
        pyramids[0].append(L1)
        # pyramids[0].reverse()
        pyramids[1].append(L2)
        # pyramids[1].reverse()
    return pyramids


# def regional_entropies(images_tuple, width, height):
#     rect = morphology.rectangle(width, height)
#     nchannels = images_tuple[0].shape[-1]
#     entropies = []
#     for img in images_tuple:
#         channelwise = []
#         for channel in range(nchannels):
#             channelwise.append(entropy(img[:, :, channel], rect))
#         entropies.append(channelwise)
#     return [entropy(img, rect) for img in images_tuple]
#


def regional_entropies(images_tuple, kernel_size):
    rect = morphology.rectangle(kernel_size, kernel_size)
    return [entropy(img, rect) for img in images_tuple]
#
#
# def regional_var(img, width, height):
#     nx = img.shape[0]
#     ny = img.shape[1]
#     vars = np.zeros((nx, ny))
#     w = width // 2
#     h = height // 2
#     for i in range(nx):
#         for j in range(ny):
#             sub = img[max(0, i - h): min(nx, i + h),
#                   max(0, j - w): min(ny, j + w)]
#             vars[i, j] = sub.var()
#     return vars
#
#
# def regional_vars(images_tuple, width, height):
#     return [regional_var(img, width, height) for img in images_tuple]


def regional_var(img, kernel_size):
    local_mean = cv2.blur(img, (kernel_size, kernel_size))
    var_im = (img.astype(np.float32) - local_mean) ** 2
    return cv2.blur(var_im, (kernel_size, kernel_size))


def regional_vars(images_tuple, kernel_size):
    return [regional_var(img, kernel_size) for img in images_tuple]


def regional_energy(img, kernel):
    img_sqr = img ** 2
    energy = cv2.filter2D(img_sqr, cv2.CV_32F, kernel)
    return energy


def regional_energies(images_tuple, kernel):
    return [regional_energy(img, kernel) for img in images_tuple]


def fuse_last_pyramid(last_pyramid_tuple, kernel_size):
    vars = regional_vars(last_pyramid_tuple, kernel_size)
    entropies = regional_entropies(last_pyramid_tuple, kernel_size)
    mask1 = (vars[0] >= vars[1]).astype(int) * (entropies[0] >= entropies[1]).astype(int)
    mask2 = (mask1 + 1).copy()
    np.place(mask2, mask2 == 2, 0)
    mask3 = 1 - (mask1 + mask2)
    return mask1 * last_pyramid_tuple[0] + mask2 * last_pyramid_tuple[1] + mask3 * 0.5 * (last_pyramid_tuple[0] + last_pyramid_tuple[1])


def fuse_pyramid_not_last(pyramid_tuple, kernel_coef=30):
    npix = pyramid_tuple[0].shape[0]
    kernel = gaussian_kernel(npix // kernel_coef)
    energies = regional_energies(pyramid_tuple, kernel)
    mask1 = np.abs(energies[0] > energies[1]).astype(int)
    mask2 = (mask1 + 1).copy()
    np.place(mask2, mask2 == 2, 0)
    return mask1 * pyramid_tuple[0] + mask2 * pyramid_tuple[1]


def fuse_pyramids(pyramids_list, kernel_size, kernel_coef):
    fused = []
    last_fused = fuse_last_pyramid((pyramids_list[0][0], pyramids_list[1][0]), kernel_size)
    fused.append(last_fused)
    levels = len(pyramids_list[0])
    for i in range(1, levels):
        fused.append(fuse_pyramid_not_last((pyramids_list[0][i], pyramids_list[1][i]), kernel_coef))
    return fused


def reconstruct(fused_pyramids):
    ls_ = np.array(fused_pyramids[0], dtype=np.uint8)
    for i in range(1, len(fused_pyramids)):
        ls_ = cv2.pyrUp(ls_)
        ls_ = cv2.add(ls_, np.array(fused_pyramids[i], dtype=np.uint8))
    return ls_






