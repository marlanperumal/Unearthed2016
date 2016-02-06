from viewer import viewer
from aggregate import aggregateData
from segmentation import process_image
import skimage.io

image = skimage.io.imread("Data/Unearthed Cape Town/De Beers Particle Size Challenge/ParticleSegmentationImages/original1.png")

image, labelledFeat = process_image(image)
colorData, sizeData = aggregateData(image,labelledFeat)

view = viewer.viewerClass(image, labelledFeat, colorData, sizeData)
view.view()