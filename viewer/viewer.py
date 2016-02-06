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

def getBinRange(bin,binNum):
    binRange = []
    binRange.append(bin[binNum])
    binRange.append(bin[binNum+1])
    return binRange

def view(imgArray, labFeat, colorData, sizeData):

    n1, binsSize, patches1 = ax3.hist(sizeData[1:], 50, facecolor='green', alpha=0.75)
    n2, binsColour, patches2 = ax4.hist(colorData[1:], 50, facecolor='green', alpha=0.75)
    addFigToAx(ax1, imgArray)
    colourArray = createWhiteArray(imgArray, labFeat)
    colourRange = getBinRange(binsSize, 18)
    colourArray = createColourArray(sizeData, labFeat, colourArray, colourRange)
    addFigToAx(ax2, colourArray)
    plt.show()

def createColourArray(adataArray, alabFeat, aimgArray, arange):
    mparticle = np.max(alabFeat)
    for ifeat in range(mparticle): # Don't look at 0 as that is the background.
        if(adataArray[ifeat] >= arange[0] and adataArray[ifeat] < arange[1]):
            aimgArray[alabFeat == ifeat] = [0,1,0]
    return aimgArray




