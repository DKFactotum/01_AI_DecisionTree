# Daniil Koshelyuk exercise on AI: Decision trees led by WelchLabs.

################################################################################
################################################################################
######## Functionality borrowed from Welch Labs for studying Purposes. #########
################################################################################
################################################################################

################################## Imports #####################################

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
from PIL import Image

################################################################################
########################### Support Functionality ##############################
################################################################################

############################ Display Examples ##################################
# Extract 9 by 9 grid and finger/not finger label from imageDict
def extractFeatures(imageDict, whichImage = 'image1bit', dist = 4):
    img = imageDict[whichImage]

    featuresList = []
    target = np.zeros(imageDict['numPointsInBox'])
    counter = 0

    for i in np.arange(imageDict['boxEdges'][2], imageDict['boxEdges'][3]):
        for j in np.arange(imageDict['boxEdges'][0], imageDict['boxEdges'][1]):
            f = img[i-dist:i+dist+1, j-dist:j+dist+1]

            fVec = f.ravel()
            featuresList.append(fVec)

            #Check and see if this is a finger pixel or not:
            if np.max(np.sum(imageDict['allFingerPoints'] == [i, j], 1)) == 2:
                target[counter] = 1

            counter = counter +1

    features = np.vstack((featuresList))

    return features, target
