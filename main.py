from viewer import viewer
from aggregate import aggregateData
from segmentation import process_image
import skimage.io
import matplotlib.pyplot as plt
import numpy as np

image = skimage.io.imread("Data/Unearthed Cape Town/De Beers Particle Size Challenge/ParticleSegmentationImages/original1.png")

image, labelledFeat = process_image(image)
colorData, sizeData = aggregateData(image,labelledFeat)

# fig, ax = plt.subplots(1,1)
# ax.imshow(labelledFeat, cmap=plt.cm.gray)
# plt.show()

view = viewer.viewerClass(image, labelledFeat, colorData, sizeData)
view.view()