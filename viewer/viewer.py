import matplotlib.pyplot as plt
import numpy as np
import skimage.io



class viewerClass:

    numBins = 20
    lastSizeSelect = None
    lastColorSelect = None

    # Setting up layout
    ax1 = plt.subplot2grid((4,4), (0,0), colspan=4)
    ax1.axis("off")
    # plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
    ax1.set_title('Original')
    ax2 = plt.subplot2grid((4,4), (1,0), colspan=4)
    ax2.axis("off")
    ax2.set_title('Processed')
    ax3 = plt.subplot2grid((4,4), (2, 0), colspan=2,rowspan=2)
    ax3.set_title('Size Histogram')
    ax4 = plt.subplot2grid((4,4), (2, 2), colspan=2,rowspan=2)
    ax4.set_title('Color Intensity Histogram')

    def __init__(self,imgArray, labFeat, colorData, sizeData):
        self.imgArray = imgArray
        self.labFeat = labFeat
        self.colorData = colorData
        self.sizeData = sizeData

    def view(self):
        self.n1, self.binsSize, self.patches1 = self.ax3.hist(self.sizeData[1:], self.numBins, facecolor='green', alpha=0.75,picker=5)
        self.n2, self.binsColour, self.patches2 = self.ax4.hist(self.colorData[1:], self.numBins, facecolor='green', alpha=0.75,picker=5)
        self.addFigToAx(self.ax1, self.imgArray)
        self.addFigToAx(self.ax2, self.createWhiteArray(self.imgArray, self.labFeat))
        self.ax3.figure.canvas.mpl_connect('pick_event', self.pick)
        plt.tight_layout()
        plt.show()

    def addFigToAx(self, aax,afig, cmap=plt.cm.gray):
        aax.imshow(afig, cmap=cmap)

    def createWhiteArray(self,imgArray, labFeat):
        whiteArr = np.zeros((imgArray.shape[0], imgArray.shape[1], 3))
        for i in np.unique(labFeat)[1:]:
            whiteArr[labFeat == i] = [1,1,1]
        return whiteArr

    def getBinRange(self, bin, binNum):
        binRange = []
        binRange.append(bin[binNum])
        binRange.append(bin[binNum+1])
        return binRange

    def createColourArray(self, adataArray, alabFeat, aimgArray, arange):
        mparticle = np.max(alabFeat)
        for ifeat in range(mparticle): # Don't look at 0 as that is the background.
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
                plt.setp(self.patches2[bin], color="r")
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


