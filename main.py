from viewer import viewer
from aggregate import aggregateData
from segmentation import process_image, trim_borders
import skimage.io
import matplotlib.pyplot as plt
import numpy as np
import os
import time
import scipy.ndimage as ndi

DIR = "Data/Unearthed Cape Town/De Beers Particle Size Challenge/Originals"
TRUTHDIR = "Data/Unearthed Cape Town/De Beers Particle Size Challenge/Truth"
for file in os.listdir(DIR):
    if file.endswith(".png"):
        FILEPATH = DIR+"/"+file
        TRUTHPATH = TRUTHDIR+"/truth"+file[-5:]
        print(FILEPATH)
        image = skimage.io.imread(FILEPATH)
        truth = skimage.io.imread(TRUTHPATH)
        border_width = 150
        image = trim_borders(image,border_width)
        image, labelledFeat, procTime = process_image(image)
        colorData, sizeData, featTime = aggregateData(image,labelledFeat)

        truth = trim_borders(truth, border_width)

        # filter truth image
        truth2 = np.zeros_like(truth)
        truth2[truth < 100] = 1
        truth2[truth > 100] = 0
        truth_particles, truth_features = ndi.label(truth2)

        view = viewer.viewerClass(image, labelledFeat, colorData, sizeData)
        view.view(selectSize=True, sizeValue=200)


image = skimage.io.imread("Data/Unearthed Cape Town/De Beers Particle Size Challenge/Originals/original1.png")
image = trim_borders(image,150)
image, labelledFeat, procTime = process_image(image)
colorData, sizeData, featTime = aggregateData(image,labelledFeat)
view = viewer.viewerClass(image, labelledFeat, colorData, sizeData)
view.view(selectSize=True, sizeValue=200)
view.show()

