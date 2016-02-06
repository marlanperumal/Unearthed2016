from viewer import viewer
from aggregate import aggregateData
import skimage.io

image = skimage.io.imread("Data/Unearthed Cape Town/De Beers Particle Size Challenge/ParticleSegmentationImages/original1.png")

colorData,SizeData = aggregateData(None,None)
viewer.view(image, image, colorData, SizeData)