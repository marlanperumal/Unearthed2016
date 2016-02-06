import numpy as np
import matplotlib.pyplot as plt
from skimage import data
from skimage.filter import canny, sobel
from scipy import ndimage as ndi
from skimage import morphology
from skimage.color import label2rgb
import skimage

image = skimage.io.imread("Data/Unearthed Cape Town/De Beers Particle Size Challenge/ParticleSegmentationImages/original1.png")
truth = skimage.io.imread("Data/Unearthed Cape Town/De Beers Particle Size Challenge/ParticleSegmentationImages/truth1.png")
# image = image[:, 500:600]
# truth = truth[:, 500:600]
truth2 = np.zeros_like(truth)
truth2[truth < 100] = 1
truth2[truth > 100] = 0

edges = canny(image/255.,2)

fill_particles = ndi.binary_fill_holes(edges)

elevation_map = sobel(image)
markers = np.zeros_like(image)
markers[image < 30] = 1
markers[image > 8000] = 2

filtered_image = np.zeros_like(image)
filtered_image[image > 8000] = 1
edges2 = canny(filtered_image/1.0)
fill_particles2 = ndi.binary_fill_holes(edges2)

segmentation = morphology.watershed(elevation_map, markers)
segmentation = ndi.binary_fill_holes(segmentation-1)
labeled_particles, num_features = ndi.label(filtered_image)
image_label_overlay = label2rgb(labeled_particles, image=image, bg_label=0)


labeled_particles2, num_features2 = ndi.label(truth2)
image_label_overlay2 = label2rgb(labeled_particles2, image=truth2, bg_label=0)
edges3 = canny(labeled_particles/float(np.max(labeled_particles)))


fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(30, 10), sharex=True, sharey=True)

ax1.imshow(edges3, cmap=plt.cm.gray)
ax1.axis("off")

ax2.imshow(image, cmap=plt.cm.gray)
ax2.axis("off")

ax3.imshow(image_label_overlay)
ax3.axis("off")

# ax4.imshow(truth2, cmap=plt.cm.gray)
# ax4.axis("off")

ax4.imshow(image_label_overlay2)
ax4.axis("off")

plt.show()
print(image.shape)
print(np.max(image))
print("Found Features",num_features)
print("Truth Features",num_features2)
