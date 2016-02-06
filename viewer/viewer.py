import matplotlib.pyplot as plt
import numpy as np
import skimage.io


numBins = 10

# Setting up layout
ax1 = plt.subplot2grid((4,4), (0,0), colspan=4)
ax1.axis("off")
ax1.set_title('Original')
ax2 = plt.subplot2grid((4,4), (1,0), colspan=4)
ax2.axis("off")
ax2.set_title('Processed')
ax3 = plt.subplot2grid((4,4), (2, 0), colspan=2,rowspan=2)
ax3.set_title('Size Histogram')
ax4 = plt.subplot2grid((4,4), (2, 2), colspan=2,rowspan=2)
ax4.set_title('Color Intensity Histogram')

def addFigToAx(aax,afig, cmap=plt.cm.gray):
    aax.imshow(afig, cmap=cmap)


def createWhiteArray(imgArray, labFeat):
    whiteArr = np.zeros((imgArray.shape[0], imgArray.shape[1], 3))
    for i in np.unique(labFeat)[1:]:
        whiteArr[labFeat == i] = [1,1,1]
    return whiteArr


def binSizeData(sizeData):
    print(sizeData)




def view(imgArray, labFeat, colorData, sizeData):

    sizeData = sizeData[1:]
    n, bins, patches = ax3.hist(sizeData, 50 ,range=[sizeData.min(),sizeData.max()], facecolor='green', alpha=0.75)
    n, bins, patches = ax4.hist(colorData, 50 ,range=[colorData.min(),colorData.max()], facecolor='green', alpha=0.75)
    print(bins)
    addFigToAx(ax1, imgArray)
    addFigToAx(ax2, createWhiteArray(imgArray, labFeat))
    plt.show()




