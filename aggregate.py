import numpy as np
def aggregateData(aimageArray, alabelledParticles):

    # Work out number of particles and size data containers
    mparticle = np.max(np.unique(alabelledParticles))
    coloData = np.zeros(mparticle)
    sizeData = np.zeros(mparticle)

    # Average colour intensity and count pixels for each particle
    for iparticle in range(mparticle):
        coloData[iparticle] = np.average(aimageArray[alabelledParticles == iparticle])
        sizeData[iparticle] = np.count_nonzero(aimageArray[alabelledParticles == iparticle])
    coloData[np.isnan(coloData)] = 0
    sizeData[np.isnan(sizeData)] = 0

    return coloData, sizeData
