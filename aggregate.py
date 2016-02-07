import numpy as np
import time
def aggregateData(aimageArray, alabelledParticles):
    tic = time.clock()
    # Work out number of particles and size data containers
    mparticle = np.max(np.unique(alabelledParticles))
    # coloData = np.zeros(mparticle)
    # sizeData = np.zeros(mparticle)
    #
    # # Average colour intensity and count pixels for each particle
    # for iparticle in range(mparticle):
    #     coloData[iparticle] = np.average(aimageArray[alabelledParticles == iparticle])
    #     sizeData[iparticle] = np.count_nonzero(aimageArray[alabelledParticles == iparticle])
    # coloData[np.isnan(coloData)] = 0
    # sizeData[np.isnan(sizeData)] = 0
    coloData = np.array([np.average(aimageArray[alabelledParticles == iparticle]) for iparticle in range(mparticle)])
    sizeData = np.array([np.count_nonzero(aimageArray[alabelledParticles == iparticle]) for iparticle in range(mparticle)])
    toc = time.clock()
    procTime = toc - tic
    return coloData, sizeData, procTime
