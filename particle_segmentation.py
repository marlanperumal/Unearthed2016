import numpy as np
import matplotlib.pyplot as plt
from skimage.filter import canny, sobel
from scipy import ndimage as ndi
from skimage import morphology
from skimage.color import label2rgb
from skimage.segmentation import felzenszwalb
import skimage.io
import skimage.color

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

# set up plot window
fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(30, 10), sharex=True, sharey=True)

# original image
ax1.imshow(image, cmap=plt.cm.gray)
ax1.axis("off")

# edges
ax2.imshow(edges, cmap=plt.cm.gray)
ax2.axis("off")

# image features
ax3.imshow(image_label_overlay)
ax3.axis("off")

# truth features
ax4.imshow(image_label_overlay2)
ax4.axis("off")

plt.show()
print("Found Features", num_features)
print("Truth Features", num_features2)

# elevation_map = sobel(image)
# markers = np.zeros_like(image)
# markers[image < 30] = 1
# markers[image > 8000] = 2

