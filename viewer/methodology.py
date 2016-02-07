from segmentation import process_image
from aggregate import aggregateData
import skimage.io
import matplotlib.pyplot as plt

image = skimage.io.imread("Data/Unearthed Cape Town/De Beers Particle Size Challenge/ParticleSegmentationImages/original3.png")
truth = skimage.io.imread("Data/Unearthed Cape Town/De Beers Particle Size Challenge/ParticleSegmentationImages/original3.png")

# trim borders
border_width = 50
truth = truth[:, border_width:-border_width]

image, labelledFeat = process_image(image)
colorData, sizeData = aggregateData(image, labelledFeat)

fig, (ax1, ax2, ax3) = plt.subplots(3,1,sharex=True,sharey=True,figsize=(30,10))

ax1.imshow(image, cmap=plt.cm.gray)
ax1.imshow(truth, cmap=plt.cm.gray)
ax1.imshow(labelledFeat, cmap=plt.cm.gray)

ax1.axis("off")
ax2.axis("off")
ax3.axis("off")