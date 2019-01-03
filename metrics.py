import numpy as np
import cv2


def get_logproba(img):
    hist = cv2.calcHist([img], [0], None, [256], [0, 256])
    loghist = np.log(hist)
    logprobas = loghist - np.log(np.sum(loghist[loghist != -np.inf]))
    return logprobas


def entropy(img):
    ent = 0
    logp = get_logproba(img)
    for i in range(255):
        if logp[i] != -np.inf:
            ent -= np.exp(logp[i]) * logp[i]
    return ent[0]


def get_joint_logprobas(img1, img2):
    hist = cv2.calcHist([img1, img2], [0, 1], None, [256, 256], [0, 256, 0, 256])
    loghist = np.log(hist)
    logprobas = loghist - np.log(np.sum(loghist[loghist != -np.inf]))
    return logprobas


def mutual_information(img1, img2):
    logjoint = get_joint_logprobas(img1, img2)
    logp1 = get_logproba(img1)
    logp2 = get_logproba(img2)
    mutualinfo = 0
    for i in range(255):
        for j in range(255):
            if (logjoint[i, j] != - np.inf) and (logp1[i] != -np.inf) and (logp2[i] != -np.inf):
                mutualinfo += np.exp(logjoint[i, j]) * (logjoint[i, j] - logp1[i] - logp2[j])
    return mutualinfo[0]


def Q_mutual_information(img1, img2, fused_img):
    hf = entropy(fused_img)
    ha = entropy(img1)
    hb = entropy(img2)
    miaf = mutual_information(img1, fused_img)
    mibf = mutual_information(img2, fused_img)
    return (2 * (miaf / (ha + hf) + mibf / (hb + hf)))


def regional_mean(img, kernel_size):
    local_mean = cv2.blur(img, (kernel_size, kernel_size))
    return local_mean


def regional_var(img, kernel_size):
    local_mean = cv2.blur(img, (kernel_size, kernel_size))
    var_im = (img.astype(np.float32) - local_mean) ** 2
    return cv2.blur(var_im, (kernel_size, kernel_size))


def regional_cov(img1, img2, kernel_size):
    mu1 = cv2.blur(img1, (kernel_size, kernel_size))
    mu2 = cv2.blur(img2, (kernel_size, kernel_size))
    covim = (img1 - mu1) * (img2 - mu2)
    return cv2.blur(covim, (kernel_size, kernel_size))


def get_ssim_mat(img1, img2, K1, K2, kernel_size=7):
    mu1 = regional_mean(img1, kernel_size)
    mu2 = regional_mean(img2, kernel_size)
    var1 = regional_var(img1, kernel_size)
    var2 = regional_var(img2, kernel_size)
    cov = regional_cov(img1, img2, kernel_size)
    C1 = (255 * K1) ** 2
    C2 = (255 * K2) ** 2
    ssim_mat = ((2 * mu1 * mu2 + C1) * (2 * cov + C2)) / ((mu1 ** 2 + mu2 ** 2 + C1) * (var1 + var2 + C2))
    lambda_mat = var1 / (var1 + var2)
    return ssim_mat, lambda_mat


def Q_ssim(img1, img2, fused_image, K1=0.01, K2=0.03, kernel_size=7):
    ssim_mat1f, lambda_mat1f = get_ssim_mat(img1, fused_image, K1, K2, kernel_size)
    ssim_mat2f, lambda_mat2f = get_ssim_mat(img2, fused_image, K1, K2, kernel_size)
    ssim_mat12, lambda_mat12 = get_ssim_mat(img2, fused_image, K1, K2, kernel_size)
    mask = (ssim_mat12 >= 0.75).astype(int)
    return np.mean(mask * (lambda_mat12 * ssim_mat1f + (1 - lambda_mat12) * ssim_mat2f) + (1 - mask) * np.maximum(ssim_mat1f, ssim_mat2f))








