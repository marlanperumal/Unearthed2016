import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons
import matplotlib.pyplot as plt
from skimage.filter import canny, sobel
from scipy import ndimage as ndi
from skimage import morphology
from skimage.color import label2rgb
from skimage.segmentation import felzenszwalb
import skimage.io

# load images
image = skimage.io.imread("Data/Unearthed Cape Town/De Beers Particle Size Challenge/ParticleSegmentationImages/original1.png")

# perform canny edge detection on image. Scale to max pixel value first
max_pixel_value = np.max(image)
thresh = 0.1*max_pixel_value

#figures
fig, (ax1) = plt.subplots(1, 1, figsize=(30, 10), sharex=True, sharey=True)

#Slider settings
axcolor = 'lightgoldenrodyellow'
axthresh = plt.axes([0.25, 0.1, 0.65, 0.03], axisbg=axcolor)
sthresh = Slider(axthresh, 'thresh', 0, max_pixel_value, valinit=thresh)

def createImage(thres):
    # do simple filter based on color value
    filtered_image = np.zeros_like(image) # set up all-zero image
    filtered_image[image > thres] = 1 # filtered values set to 1

    # label features and convert to rgb image
    labeled_particles, num_features = ndi.label(filtered_image)
    image_label_overlay = label2rgb(labeled_particles, image=image, bg_label=0)
    return image_label_overlay

def updatePlot(value):
    thresh = sthresh.val
    newImage = createImage(thresh)
    ax1.imshow(newImage, cmap=plt.cm.gray)
    ax1.axis("off")
    fig.canvas.draw_idle()

updatePlot(thresh)
sthresh.on_changed(updatePlot)
plt.show()