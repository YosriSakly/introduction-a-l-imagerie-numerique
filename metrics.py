import numpy as np

def get_proba(img):
    hist = np.zeros(255)
    npixs = img.shape[0] * img.shape[1]
    for i in range(0, 255):
        hist[i] = np.sum(img == i) / npixs
    return hist

def entropy(proba_vec):
    return -np.sum(np.log(proba_vec) * proba_vec)


def get_joint_probas(img1, img2):



def mutual_information(img1, img2):

