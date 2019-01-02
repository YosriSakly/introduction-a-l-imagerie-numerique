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
    return (2 * (miaf / (ha + hf) + mibf / (hb + hf)))[0]





