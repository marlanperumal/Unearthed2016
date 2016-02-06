import numpy as np
import matplotlib.pyplot as plt
from skimage.filter import canny, sobel
from skimage.filter import threshold_otsu as threshold_func
from skimage.feature import peak_local_max
from scipy import ndimage as ndi
from skimage import morphology
from skimage.color import label2rgb
from skimage.segmentation import felzenszwalb
import skimage.io
import skimage.color
from skimage.segmentation import join_segmentations

# load images
image = skimage.io.imread("Data/Unearthed Cape Town/De Beers Particle Size Challenge/ParticleSegmentationImages/original1.png")
truth = skimage.io.imread("Data/Unearthed Cape Town/De Beers Particle Size Challenge/ParticleSegmentationImages/truth1.png")

# trim borders
border_width = 50
image = image[:, border_width:-border_width]
truth = truth[:, border_width:-border_width]

# image = image[0:40,370:450]
# truth = truth[0:40,370:450]

# perform canny edge detection on image. Scale to max pixel value first
max_pixel_value = np.max(image)
edges = canny(image/float(max_pixel_value), sigma=1.0)

# do simple filter based on color value
# thresh = 0.08*max_pixel_value
thresh = threshold_func(image)
filtered_image = np.zeros_like(image,dtype=np.uint8) # set up all-zero image
filtered_image[image > thresh] = 1 # filtered values set to 1
filtered_image1 = ndi.binary_fill_holes(filtered_image)

elevation_map = sobel(image)
distance = ndi.distance_transform_edt(filtered_image)
dist_thresh = 0.1
dist_kernals = np.zeros_like(distance)
dist_kernals[distance > dist_thresh*np.max(distance)] = 1

dilated = morphology.dilation(filtered_image, morphology.square(3))
eroded = morphology.erosion(filtered_image, morphology.square(3))

# local_maxi = peak_local_max(distance, indices=False, footprint=np.ones((3, 3)),
#                             labels=filtered_image)
# markers = ndi.label(local_maxi)[0]
# labels = morphology.watershed(-distance, markers, mask=filtered_image)
# print(np.unique(labels).size)

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
# edges3 = canny(labeled_particles/float(np.max(labeled_particles)))

# segmentations = []
#
# for thresh_factor in np.arange(0.05,0.25,0.01):
#     # do simple filter based on color value
#     thresh = thresh_factor*max_pixel_value
#     filtered_image = np.zeros_like(image) # set up all-zero image
#     filtered_image[image > thresh] = 1 # filtered values set to 1
#
#     # label features and convert to rgb image
#     labeled_particles, num_features = ndi.label(filtered_image)
#     segmentations.append(labeled_particles)
#     print(thresh_factor, num_features)
#
# segmentation_join = segmentations[0]
# for segmentation in segmentations:
#     segmentation_join = join_segmentations(segmentation_join,segmentation)
# segmentation_join = label2rgb(segmentation_join, image=image, bg_label=0)

# set up plot window
fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(30, 10), sharex=True, sharey=True)

# original image
ax1.imshow(image, cmap=plt.cm.gray)
ax1.axis("off")

# edges
ax2.imshow(filtered_image, cmap=plt.cm.gray)
ax2.axis("off")

# image features
ax3.imshow(dilated, cmap=plt.cm.gray)
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

