import os.path
import utils


# Path to the data
path = os.getcwd() + "/Data/rose"

# Load images
images = utils.load_images(path, gray=True, align=False)

# Size of kernel for basis / details decomposition
avg_size=31
# Laplacian filter size
lapsize = 1
# Parameters for Gaussian lowpass filter
rg = 5
sigmag = 5
# Refined weight maps using guided filtering parameters
eps1 = 0.3
r1 = 45
eps2 = 1e-5
r2 =7

fused = utils.fuse_images(images, avg_size, lapsize, rg, sigmag, eps1, r1, eps2, r2, plot=True)




