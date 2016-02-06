import numpy as np
import matplotlib.pyplot as plt
from skimage.filter import canny, sobel
from scipy import ndimage as ndi
from skimage import morphology
from skimage.color import label2rgb
from skimage.segmentation import felzenszwalb
from skimage.segmentation import*
import skimage.io
import skimage.color
from skimage.morphology import watershed
from skimage.feature import peak_local_max

# load images
image = skimage.io.imread("Data/Unearthed Cape Town/De Beers Particle Size Challenge/ParticleSegmentationImages/original1.png")
truth = skimage.io.imread("Data/Unearthed Cape Town/De Beers Particle Size Challenge/ParticleSegmentationImages/truth1.png")

# trim borders
border_width = 50
image = image[:, border_width:-border_width]
truth = truth[:, border_width:-border_width]

# perform canny edge detection on image. Scale to max pixel value first
max_pixel_value = np.max(image)
edges = canny(image/float(max_pixel_value), sigma=1.0)

# do simple filter based on color value
thresh = 0.1*max_pixel_value
filtered_image = np.zeros_like(image) # set up all-zero image
filtered_image[image > thresh] = 1 # filtered values set to 1

# label features and convert to rgb image
labeled_particles, num_features = ndi.label(filtered_image)
image_label_overlay = label2rgb(labeled_particles, image=image, bg_label=0)

# filter truth image
truth2 = np.zeros_like(truth)
truth2[truth < 100] = 1
truth2[truth > 100] = 0

# label truth features
labeled_particles2, num_features2 = ndi.label(truth2)
image_label_overlay2 = label2rgb(labeled_particles2, image=truth2, bg_label=0)
edges3 = canny(labeled_particles/float(np.max(labeled_particles)))

# Aggregate data
coloData = np.zeros(num_features)
sizeData = np.zeros(num_features)

for ifeat in range(num_features):
    coloData[ifeat] = np.average(image[labeled_particles == ifeat])
    sizeData[ifeat] = np.count_nonzero(filtered_image[labeled_particles == ifeat])

