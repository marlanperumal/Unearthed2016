import matplotlib.pyplot as plt
import skimage.io

# Dummy image for now
image = skimage.io.imread("../Data/Unearthed Cape Town/De Beers Particle Size Challenge/ParticleSegmentationImages/original1.png")
truth = skimage.io.imread("../Data/Unearthed Cape Town/De Beers Particle Size Challenge/ParticleSegmentationImages/truth1.png")

# Setting up layout
ax1 = plt.subplot2grid((4,4), (0,0), colspan=4)
ax1.axis("off")
ax1.set_title('Original')
ax2 = plt.subplot2grid((4,4), (1,0), colspan=4)
ax2.axis("off")
ax2.set_title('Processed')
ax3 = plt.subplot2grid((4,4), (2, 0), colspan=2,rowspan=2)
ax4 = plt.subplot2grid((4,4), (2, 2), colspan=2,rowspan=2)

def addFigToAx(aax,afig):
    aax.imshow(afig, cmap=plt.cm.gray)

def view(imgArray, labFeat, colorData, sizeData):
    addFigToAx(ax1, imgArray)
    addFigToAx(ax2, truth)



plt.show()
