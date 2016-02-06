import numpy as np
from skimage.io import imread
import scipy.ndimage as ndi
import matplotlib.pyplot as plt
from skimage.color import label2rgb

def process_image(image_array):
    image = imread("Data/Unearthed Cape Town/De Beers Particle Size Challenge/ParticleSegmentationImages/original1.png")
    truth = imread("Data/Unearthed Cape Town/De Beers Particle Size Challenge/ParticleSegmentationImages/truth1.png")

    # trim borders
    border_width = 50
    image = image[:, border_width:-border_width]
    truth = truth[:, border_width:-border_width]

    # filter truth image
    truth2 = np.zeros_like(truth)
    truth2[truth < 100] = 1
    truth2[truth > 100] = 0

    # label truth features
    labeled_particles, num_features = ndi.label(truth2)

    return image, labeled_particles

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



