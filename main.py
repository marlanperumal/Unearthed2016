from viewer import viewer
from aggregate import aggregateData
from segmentation import process_image
import skimage.io
import matplotlib.pyplot as plt
import numpy as np
import os
DIR = "Data/Unearthed Cape Town/De Beers Particle Size Challenge/Originals"
for file in os.listdir(DIR):
    if file.endswith(".png"):
        FILEPATH = DIR+"/"+file
        print(FILEPATH)
        image = skimage.io.imread(FILEPATH)
        image, labelledFeat = process_image(image)
        colorData, sizeData = aggregateData(image,labelledFeat)
        view = viewer.viewerClass(image, labelledFeat, colorData, sizeData)
        view.view()
