from viewer import viewer
from aggregate import aggregateData
from segmentation import process_image, trim_borders
import skimage.io
import matplotlib.pyplot as plt
import numpy as np
import os
import time

DIR = "Data/Unearthed Cape Town/De Beers Particle Size Challenge/Originals"
for file in os.listdir(DIR):
    if file.endswith(".png"):
        FILEPATH = DIR+"/"+file
        print(FILEPATH)
        image = skimage.io.imread(FILEPATH)
        image = trim_borders(image,150)
        image, labelledFeat = process_image(image)
        colorData, sizeData = aggregateData(image,labelledFeat)
        view = viewer.viewerClass(image, labelledFeat, colorData, sizeData)
        view.view(selectSize=True, sizeValue=200, acont=True)


image = skimage.io.imread("Data/Unearthed Cape Town/De Beers Particle Size Challenge/Originals/original1.png")
image = trim_borders(image,150)
image, labelledFeat = process_image(image)
colorData, sizeData = aggregateData(image,labelledFeat)
view = viewer.viewerClass(image, labelledFeat, colorData, sizeData)
view.view(selectSize=True, sizeValue=200,acont=False)
view.show()

