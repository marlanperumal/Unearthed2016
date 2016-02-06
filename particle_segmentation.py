import numpy as np
import matplotlib.pyplot as plt
from skimage.filter import canny, sobel
from skimage.filter import threshold_otsu as threshold_func
from skimage.feature import peak_local_max
from scipy import ndimage as ndi
from skimage import morphology
from skimage.color import label2rgb
import skimage.io
import skimage.color
from skimage.exposure import rescale_intensity
import time
from skimage.segmentation import find_boundaries

def apply_watershed(image, feature_num, tot_features):
    # print(image)
    feature_mask = np.zeros((image.shape[0]+6,image.shape[1]+6))
    feature_mask[3:-3,3:-3][image == feature_num] = 1
    distance = ndi.distance_transform_edt(feature_mask)
    # distance[distance > 0.7*np.max(distance)] = 0.7*np.max(distance)
    template_size = 5
    # if np.sum(feature_mask) < 100:
    #     template_size = 3
    # else:
    #     template_size = 5
    local_maxi = peak_local_max(distance, indices=False, footprint=morphology.square(template_size),
                                labels=feature_mask)
    markers = ndi.label(local_maxi)[0]
    # labels = morphology.watershed(-distance, markers, mask=feature_mask)
    labels = morphology.watershed(-distance, markers, mask=feature_mask)
    # plt.imshow(markers, cmap=plt.cm.gray, interpolation="nearest")
    # plt.show()
    # image[labels[1:-1,1:-1] == 1] = feature_num

    for i in np.unique(labels)[2:]:
        tot_features = tot_features + 1
        image[labels[3:-3,3:-3] == i] = tot_features
    return tot_features

tic = time.clock()
# load images

image = skimage.io.imread("Data/Unearthed Cape Town/De Beers Particle Size Challenge/ParticleSegmentationImages/original3.png")
truth = skimage.io.imread("Data/Unearthed Cape Town/De Beers Particle Size Challenge/ParticleSegmentationImages/truth3.png")

# trim borders
border_width = 50
image = image[:, border_width:-border_width]
truth = truth[:, border_width:-border_width]

p2, p98 = np.percentile(image, (1, 99.9))
image = rescale_intensity(1.0*image, in_range=(p2, p98))

# image = image[10:70,55:110]
# truth = truth[10:70,55:110]

# image = image[0:30,370:430]
# truth = truth[0:30,370:430]

# perform canny edge detection on image. Scale to max pixel value first
max_pixel_value = np.max(image)
edges = canny(image/float(max_pixel_value), sigma=1.0)

# do simple filter based on color value
thresh = 0.5*threshold_func(image)
filtered_image = np.zeros_like(image,dtype=np.uint8) # set up all-zero image
filtered_image[image > thresh] = 1 # filtered values set to 1
filtered_image1 = ndi.binary_fill_holes(edges)

distance = ndi.distance_transform_edt(filtered_image)

# label features and convert to rgb image
labeled_particles, num_features = ndi.label(filtered_image)
# labeled_slices = ndi.find_objects(labeled_particles)
#
# tot_features = num_features
# for i in range(num_features):
#     tot_features = apply_watershed(labeled_particles[labeled_slices[i]],i+1, tot_features)

image_label_overlay = label2rgb(labeled_particles, bg_label=0)
# clear_border(filtered_image)

new_image = image.copy()
new_image[labeled_particles == 0] = 0
# distance = ndi.distance_transform_edt(sobel(new_image))
distance = ndi.distance_transform_edt(filtered_image)
# distance[distance > 0.7*np.max(distance)] = 0.7*np.max(distance)
local_maxi = peak_local_max(distance, indices=False, footprint=morphology.square(5),
                            labels=filtered_image, exclude_border=False)
markers = ndi.label(local_maxi)[0]
# plt.imshow(distance, cmap=plt.cm.gray, interpolation="nearest")
# plt.show()

# labels = morphology.watershed(-distance, markers, mask=filtered_image)
closed_elevation = ndi.grey_closing(sobel(new_image),size=4)
labels = morphology.watershed(closed_elevation, markers)
image_label_overlay3 = label2rgb(labels, bg_label=0)
image_label_overlay3[find_boundaries(labels)] = [0,0,0]

# filter truth image
truth2 = np.zeros_like(truth)
truth2[truth < 100] = 1
truth2[truth > 100] = 0

# label truth features
labeled_particles2, num_features2 = ndi.label(truth2)
image_label_overlay2 = label2rgb(labeled_particles2, bg_label=0)

# set up plot window
fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(30, 10), sharex=True, sharey=True)

# original image
ax1.imshow(closed_elevation, cmap=plt.cm.gray)
ax1.axis("off")

# edges
ax2.imshow(image_label_overlay)
ax2.axis("off")

new_image = image.copy()
new_image[labeled_particles == 0] = 0

# image features
ax3.imshow(image_label_overlay3)
ax3.axis("off")

# truth features
ax4.imshow(image_label_overlay2)
ax4.axis("off")
toc = time.clock()
print(toc-tic)
plt.show()
print("Found Features", num_features)
print("Found Features Watershed", len(np.unique(labels)))
print("Truth Features", num_features2)



