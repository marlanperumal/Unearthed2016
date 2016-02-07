import numpy as np
from skimage.io import imread
import scipy.ndimage as ndi
import matplotlib.pyplot as plt
from skimage.color import label2rgb
from skimage.exposure import rescale_intensity
from skimage.filter import threshold_otsu as threshold_func
from skimage.feature import peak_local_max
from skimage.segmentation import find_boundaries
from skimage import morphology

def trim_borders(image, border_width=50):
    # trim borders
    return image[:, border_width:-border_width]

def process_image(image):
    # rescale intensity
    p2, p98 = np.percentile(image, (1, 99.9))
    image = rescale_intensity(1.0*image, in_range=(p2, p98))

    # do simple filter based on color value
    thresh = 0.5*threshold_func(image)
    filtered_image = np.zeros_like(image,dtype=np.uint8) # set up all-zero image
    filtered_image[image > thresh] = 1 # filtered values set to 1

    # perform watershed transform to split clusters
    distance = ndi.distance_transform_edt(filtered_image)
    local_maxi = peak_local_max(distance, indices=False, footprint=morphology.square(7),
                            labels=filtered_image, exclude_border=False)
    markers = ndi.label(local_maxi)[0]

    # segment and label particles
    labels = morphology.watershed(-distance, markers, mask=filtered_image)
    backup_labels = labels.copy()

    # remove boundaries and restore any small particles deleted in this process
    labels[find_boundaries(labels)] = 0
    for i in np.unique(backup_labels)[1:]:
        if np.count_nonzero(labels[backup_labels == i]) == 0:
            labels[backup_labels == i] = i

    return image, labels

if __name__ == "__main__":
    image = imread("Data/Unearthed Cape Town/De Beers Particle Size Challenge/ParticleSegmentationImages/original1.png")
    truth = imread("Data/Unearthed Cape Town/De Beers Particle Size Challenge/ParticleSegmentationImages/truth1.png")

    # trim borders
    border_width = 50
    image = image[:, border_width:-border_width]
    truth = truth[:, border_width:-border_width]

    _, labeled_particles = process_image(image)
    print(len(np.unique(labeled_particles)))
    image_label_overlay = label2rgb(labeled_particles, image=image, bg_label=0)
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(30, 10), sharex=True, sharey=True)
    ax1.imshow(image, cmap=plt.cm.gray)
    ax2.imshow(image_label_overlay)
    ax3.imshow(truth, cmap=plt.cm.gray)
    ax1.axis("off")
    ax2.axis("off")
    ax3.axis("off")
    plt.show()


