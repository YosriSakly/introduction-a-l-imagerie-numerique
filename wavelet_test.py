import cv2
import os
import pywt
import numpy as np
import matplotlib.pyplot as plt


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

images_float = tuple([(1 / 255) * image for image in images])


def channel_wavelets_2d(images_tuple, channel=0, wavelet_type='bior1.3'):
    LLs = []
    LHs = []
    HLs = []
    HHs = []
    for image in images_tuple:
        LL, (LH, HL, HH) = pywt.dwt2(image[:, :, channel], wavelet_type)
        LLs.append(LL)
        LHs.append(LH)
        HLs.append(HL)
        HHs.append(HH)
    return tuple(LLs), tuple(LHs), tuple(HLs), tuple(HHs)


def channel_wavelets_multi(images_tuple, channel=0, level = 5, wavelet_type='bior1.3'):
    coeffs_list = []
    for image in images_tuple:
        coeffs_list.append(pywt.wavedec2(image[:, :, channel], wavelet_type, level=level))
    LLs = [coeffs[0] for coeffs in coeffs_list]
    sec_coeffs_tuple_list = []
    for l in range (0, level):
        first = [coeffs[l][0] for coeffs in coeffs_list]
        second = [coeffs[l][1] for coeffs in coeffs_list]
        third = [coeffs[l][2] for coeffs in coeffs_list]
        sec_coeffs_tuple_list
    return tuple(LLs), tuple(LHs), tuple(HLs), tuple(HHs)



def fused_coefficients(coefs):
    coefs_array = np.array(coefs)
    masks = np.zeros(coefs_array.shape, dtype=int)
    argmax_abs = np.argmax(np.abs(coefs_array), axis=0)
    for i in range(0, coefs_array.shape[0]):
        masks[i, :, :] = (argmax_abs == i).astype(int)
    return np.sum(coefs_array * masks, axis=0)


def fuse_wavelet_channel_2d(LLs, LHs, HLs, HHs, wavelet_type='bior1.3'):
    LLmax = fused_coefficients(LLs)
    LHmax = fused_coefficients(LHs)
    HLmax = fused_coefficients(HLs)
    HHmax = fused_coefficients(HHs)
    max_coeffs2 = LLmax, (LHmax, HLmax, HHmax)
    fused = pywt.idwt2(max_coeffs2, wavelet_type)
    return fused


def fuse_images_wavelets_2d(images_tuple, wavelet_type='bior1.3'):
    shape = images_tuple[0].shape
    fused_image = np.zeros(shape)
    for channel in [0, 1, 2]:
        LLs, LHs, HLs, HHs = channel_wavelets_2d(images_tuple, channel, wavelet_type)
        fused = fuse_wavelet_channel_2d(LLs, LHs, HLs, HHs, wavelet_type)
        fused_image[:, :, channel] = fused
    return fused_image


fused_test = fuse_images_wavelets_2d(images)
fused_test = np.around(fused_test).astype(np.uint8)


LL1, (LH1, HL1, HH1) = pywt.dwt2(images[0][:, :, 0], wavelet="bior1.3")
LL2, (LH2, HL2, HH2) = pywt.dwt2(images[1][:, :, 0], wavelet="bior1.3")
test = LL1, (LH1, HL1, HH1)
dd = pywt.idwt2(test, "bior1.3")

test = pywt.wavedec2(images[0][:, :, 0], "bior1.3", level=3)