import matplotlib.pyplot as plt
import numpy as np
import skimage.io
from time import sleep
from scipy import eye

mng = plt.get_current_fig_manager()
mng.resize(*mng.window.maxsize())

class viewerClass:

    numBins = 20
    lastSizeSelect = None
    lastColorSelect = None

    def __init__(self,imgArray, labFeat, colorData, sizeData):
        self.imgArray = imgArray
        self.labFeat = labFeat
        self.colorData = colorData
        self.sizeData = sizeData
        self.isColor = []
        self.isSize = []

    def setSizeSelect(self,size):
        binNum = self.whichBin(self.binsSize, size)
        self.changeSizeImg(binNum)
        plt.setp(self.patches1[binNum], color="r")
        self.ax3.figure.canvas.draw()

    def setColorSelect(self,inten):
        binNum = self.whichBin(self.binsSize, inten)
        self.changeColorImg(binNum)
        plt.setp(self.patches2[binNum], color="r")
        self.ax4.figure.canvas.draw()

    def view(self,selectColor = False,selectSize = False, sizeValue = 0.0,colorValue = 0.0):
        self.initView()
        self.n1, self.binsSize, self.patches1 = self.ax3.hist(self.sizeData[1:], self.numBins, facecolor='green', alpha=0.75,picker=5)
        self.n2, self.binsColour, self.patches2 = self.ax4.hist(self.colorData[1:], self.numBins, facecolor='green', alpha=0.75,picker=5)
        self.addFigToAx(self.ax1, self.imgArray)
        self.addFigToAx(self.ax2, self.createWhiteArray(self.imgArray, self.labFeat))
        self.ax3.figure.canvas.mpl_connect('pick_event', self.pick)

        if(selectColor):
            self.setColorSelect(colorValue)
        if(selectSize):
            self.setSizeSelect(sizeValue)

        plt.show(block=False)

    def show(self):
        plt.show()

    def initView(self):
            # Setting up layout
        self.ax1 = plt.subplot2grid((4,4), (0,0), colspan=4)
        self.ax1.axis("off")
        self.ax1.set_title('Original')
        self.ax2 = plt.subplot2grid((4,4), (1,0), colspan=4)
        self.ax2.axis("off")
        self.ax2.set_title('Processed')
        self.ax3 = plt.subplot2grid((4,4), (2, 0), colspan=2,rowspan=2)
        self.ax3.set_title('Size Histogram')
        self.ax4 = plt.subplot2grid((4,4), (2, 2), colspan=2,rowspan=2)
        self.ax4.set_title('Color Intensity Histogram')
        plt.subplots_adjust(left=0.05, bottom=0.06, right=0.95, top=0.95, wspace=0.2, hspace=0.2)

    def addFigToAx(self, aax,afig, cmap=plt.cm.gray):
        aax.imshow(afig, cmap=cmap)


    def createWhiteArray(self,imgArray, labFeat):
        whiteArr = np.zeros((imgArray.shape[0], imgArray.shape[1], 3))
        for i in np.unique(labFeat)[1:]:
            whiteArr[labFeat == i] = [1,1,1]
        return whiteArr

    # Determines which bin the number is within
    def whichBin(self,bin,anum):
        for i in range(len(bin)):
            binBounds = self.getBinRange(bin,i)
            if(anum >= binBounds[0]  and  anum < binBounds[1]):
                return i

    def getBinRange(self, bin, binNum):
        binRange = []
        binRange.append(bin[binNum])
        binRange.append(bin[binNum+1])
        return binRange


    def createColourArray(self, adataArray, alabFeat, aimgArray, arange):
        mparticle = np.max(np.unique(alabFeat))
        for ifeat in np.unique(alabFeat)[1:]-1: # Don't look at 0 as that is the background.
            if(adataArray[ifeat] >= arange[0] and adataArray[ifeat] < arange[1]):
                aimgArray[alabFeat == ifeat] = [1.,0,0]
        return aimgArray


    def pick(self,event):

        if(event.mouseevent.inaxes == self.ax3):
            bin = self.findBinIndex(event.artist.xy[0],self.binsSize)
            if(event.artist.xy[0] == self.lastSizeSelect):
                self.addFigToAx(self.ax2, self.createWhiteArray(self.imgArray, self.labFeat))
                self.ax2.figure.canvas.draw()
                self.lastSizeSelect = None
                plt.setp(self.patches1[bin], color="g")
                self.ax3.figure.canvas.draw()
            else:
                self.changeSizeImg(bin)
                self.lastSizeSelect = event.artist.xy[0]
                plt.setp(self.patches1[bin], color="r")
                self.ax3.figure.canvas.draw()
        elif(event.mouseevent.inaxes == self.ax4):
            bin = self.findBinIndex(event.artist.xy[0],self.binsColour)
            if(event.artist.xy[0] == self.lastColorSelect):
                self.addFigToAx(self.ax2, self.createWhiteArray(self.imgArray, self.labFeat))
                self.ax2.figure.canvas.draw()
                self.lastColorSelect = None
                plt.setp(self.patches2[bin], color="g")
                self.ax3.figure.canvas.draw()
            else:
                self.changeColorImg(bin)
                self.lastColorSelect = event.artist.xy[0]
                plt.setp(self.patches2[bin], color="r")
                self.ax3.figure.canvas.draw()

    def changeSizeImg(self, bin):
        whiteArr = self.createWhiteArray(self.imgArray, self.labFeat)
        colourRange = self.getBinRange(self.binsSize, bin)
        colourArray = self.createColourArray(self.sizeData, self.labFeat, whiteArr, colourRange)
        self.ax2.cla()
        self.addFigToAx(self.ax2, colourArray)
        self.ax2.axis("off")
        self.ax2.set_title('Processed')
        self.ax2.figure.canvas.draw()

    def changeColorImg(self,bin):
        whiteArr = self.createWhiteArray(self.imgArray, self.labFeat)
        colourRange = self.getBinRange(self.binsColour, bin)
        colourArray = self.createColourArray(self.colorData, self.labFeat, whiteArr, colourRange)
        self.ax2.cla()
        self.addFigToAx(self.ax2, colourArray)
        self.ax2.axis("off")
        self.ax2.set_title('Processed')
        self.ax2.figure.canvas.draw()

    def findBinIndex(self,anum,abin):
        i = 0
        tol = 1e-3
        for index in range(len(abin)):
            if (anum-abin[index]) <= tol:
                return i
            else:
                i = i + 1