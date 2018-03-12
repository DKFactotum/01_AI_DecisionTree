# Daniil Koshelyuk exercise on AI: Decision trees led by WelchLabs.

################################################################################
################################################################################
######## Functionality borrowed from Welch Labs for studying Purposes. #########
################################################################################
################################################################################

################################## Imports #####################################

import os#, cv2
import cPickle as pickle
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

################################################################################
############################ DATA PREPROCESSING ################################
################################################################################

################################# General ######################################
# Custom function to identify unique rows in a 2d numpy arrray.
def uniqueRows(a):
    b = np.ascontiguousarray(a).view(np.dtype((np.void, a.dtype.itemsize * a.shape[1])))
    _, idx = np.unique(b, return_index=True)

    aUnique = a[idx].astype('int')

    return aUnique
# Custom function to help identify finger pixels
def expandByOne(allPoints, img, relativeThresh=0.5):
    for point in allPoints:
        #There are 8 possible search directions:
        for i in [-1,0,1]:
            for j in [-1,0,1]:
                newPoint = point + [i, j]
                #Make sure point is actually new:
                if sum(newPoint==point) == 0:
                    if img[newPoint[0], newPoint[1]] > relativeThresh*img[point[0], point[1]]:
                        allPoints = np.vstack(([allPoints, newPoint]))

    return uniqueRows(allPoints)

############################## Pickle Handling #################################
# A function to import data.
def importPickles():
    #Let's do a list of dicts
    imageDicts = []
    path = 'data/rawData/'
    for (dirpath, dirnames, filenames) in os.walk(path):
        for filename in filenames:
            if filename.endswith('.pickle'):
                pickleFileName = path + filename
                pickleFile = open(pickleFileName, 'rb')
                imageDict = pickle.load(pickleFile)
                pickleFile.close()

                imageDicts.append(imageDict)

    print 'Loaded ' + str(len(imageDicts)) + ' pickles!'
    return imageDicts
# A function to export Labled data.
def exportPickles(imageDictsLabeled, type=0):
    if (type == 0):     # All Data Export
        pickleName = 'fingerDataSet'
        directory = 'data/'
        pickleFileName = directory + pickleName + ".pickle"
        pickleFile = open(pickleFileName, 'wb')
        pickle.dump(imageDictsLabeled, pickleFile, pickle.HIGHEST_PROTOCOL)
        pickleFile.close()
    elif (type == 1):   # Reduced Data Export
        pickleName = 'fingers'
        directory = 'data/'
        pickleFileName = directory + pickleName + ".pickle"
        pickleFile = open(pickleFileName, 'wb')
        pickle.dump((fingers, notFingers), pickleFile, pickle.HIGHEST_PROTOCOL)
        pickleFile.close()

############################### Finger Finding #################################
# Find fingers.
def findFingerPoints(imageDict, fingerIndex='1', expansionIterations=4):
    tipIndices = imageDict['trackingIndices'][fingerIndex]['tipIndices']
    baseIndices = imageDict['trackingIndices'][fingerIndex]['baseIndices']

    tipMean = [np.mean(tipIndices[0]), np.mean(tipIndices[1])]
    baseMean = [np.mean(baseIndices[0]), np.mean(baseIndices[1])]

    if ~isnan(tipMean[0]) and ~isnan(baseMean[0]):
        #Fit line between means:
        m = (baseMean[1]-tipMean[1])/(baseMean[0]-tipMean[0])
        b = baseMean[1]-m*baseMean[0]

        x = linspace(baseMean[0], tipMean[0], 100)
        y = m*x +b

        points = np.vstack(([x.round(), y.round()])).T
        uniquePoints = uniqueRows(points)

        for i in range(expansionIterations):
            uniquePoints = expandByOne(allPoints = uniquePoints, img = imageDict['image'], \
                                       relativeThresh = 0.5)
        return uniquePoints
    else:
        return [0]
# Label data on images.
def labelImageDicts(imageDicts):
    imageDictsLabeled = []
    for imageDict in imageDicts:
        success = True

        #Find Index Finger:
        indexFingerPoints = findFingerPoints(imageDict, fingerIndex = '1')
        if len(indexFingerPoints)<10:
            success = False
        else:
            imageDict['indexFingerPoints'] = indexFingerPoints
            imageDict['allFingerPoints'] = indexFingerPoints

        #If Present, Find Middle Finger:
        if imageDict['numFingers']>1 and success:
            middleFingerPoints = findFingerPoints(imageDict, fingerIndex = '2')
            if len(middleFingerPoints)<10:
                success = False
            else:
                imageDict['middleFingerPoints'] = middleFingerPoints
                imageDict['allFingerPoints'] = np.vstack((imageDict['allFingerPoints'], \
                                                         middleFingerPoints))

        #If Present, Find Ring Finger:
        if imageDict['numFingers']>2 and success:
            ringFingerPoints = findFingerPoints(imageDict, fingerIndex = '3')
            if len(ringFingerPoints)<10:
                success = False
            else:
                imageDict['ringFingerPoints'] = ringFingerPoints
                imageDict['allFingerPoints'] = np.vstack((imageDict['allFingerPoints'], \
                                                 ringFingerPoints))

        if success:
            imageDictsLabeled.append(imageDict)

    print str(len(imageDictsLabeled)) + ' successes, ' + str(len(imageDicts)- len(imageDictsLabeled)) + ' failures.'

    return imageDictsLabeled

########################### Bounding Box Generation ############################
# Cover Hand
## Find all pixels in hand by expanding-by-one further:.
def expandToCoverHand(imageDict, threshold=0.8, numIterations=30):
    expandedPoints = imageDict['allFingerPoints']
    for i in range(numIterations):
        expandedPoints = expandByOne(allPoints = expandedPoints, img = imageDict['image'], \
                               relativeThresh = threshold)
    return expandedPoints
## Cover the hand pixels.
def coverHand(imageDictsLabeled):
    #Slow!
    for imageDict in imageDictsLabeled:
        expandedPoints = expandToCoverHand(imageDict, threshold = 0.8, numIterations = 30)
        imageDict['handPoints'] = expandedPoints
    return imageDictsLabeled
# Bounding Box
## Generate Bounding Boxes
def boundingBox(imageDictsLabeled, boxBorderOffset=8):
    for imageDict in imageDictsLabeled:
        minY = min(imageDict['handPoints'][:,0])
        minX = min(imageDict['handPoints'][:,1])
        maxY = max(imageDict['handPoints'][:,0])
        maxX = max(imageDict['handPoints'][:,1])

        xRange = arange(minX-boxBorderOffset, maxX+1+boxBorderOffset)
        yRange = arange(minY-boxBorderOffset, maxY+1+boxBorderOffset)

        top     = np.vstack(([(minY-boxBorderOffset)*np.ones(len(xRange)), xRange])).T
        bottom  = np.vstack(([(maxY+boxBorderOffset)*np.ones(len(xRange)), xRange])).T
        left    = np.vstack(([yRange, (minX-boxBorderOffset)*np.ones(len(yRange))])).T
        right   = np.vstack(([yRange, (maxX+boxBorderOffset)*np.ones(len(yRange))])).T

        box = np.vstack(([top, bottom, left, right]))

        imageDict['handEdges'] = [minX, maxX, minY, maxY]
        imageDict['boxEdges'] = [minX-boxBorderOffset, maxX+boxBorderOffset, minY-boxBorderOffset, maxY+boxBorderOffset]
        boxHeight = ((maxY+boxBorderOffset)- (minY-boxBorderOffset))
        boxWidth = ((maxX+boxBorderOffset)- (minX-boxBorderOffset))

        imageDict['boxHeight'] = boxHeight
        imageDict['boxWidth'] = boxWidth
        imageDict['numPointsInBox'] = boxHeight*boxWidth

        imageDict['box'] = box
    return imageDictsLabeled

############################ Picture Simplification ############################
# Reduce complexity by thresholding.
def reduceBits(imageDictsLabeled, type=0):
    for imageDict in imageDictsLabeled:
        image = imageDict['image']
        if (type == 0):     # 1 bit
            imageDict['image1bit'] = image>92
        elif (type == 1):   # 2 bits
            image2bit = np.zeros((image.shape))
            image2bit[np.logical_and(image > 64, image< 128)] = 1
            image2bit[np.logical_and(image >= 128, image< 192)] = 2
            image2bit[image>=192] = 3
            imageDict['image2bit'] = image2bit
        elif (type == 2):   # 3 bits
            image3bit = np.zeros((image.shape))
            image3bit[np.logical_and(image > 32, image< 64)] = 1
            image3bit[np.logical_and(image >= 64, image< 96)] = 2
            image3bit[np.logical_and(image >= 96, image< 128)] = 3
            image3bit[np.logical_and(image >= 128, image< 160)] = 4
            image3bit[np.logical_and(image >= 160, image< 192)] = 5
            image3bit[np.logical_and(image >= 192, image< 224)] = 6
            image3bit[image>=224] = 7
            imageDict['image3bit'] = image3bit
        else:
            pass
    return imageDictsLabeled

############################## Cropped Data ####################################
# Arrange Dictionary with Cropped images.
def arrangeCroppedImages(imageDictsLabeled):
    for i in range(len(imageDictsLabeled)):
        imageDictsLabeled[i]['croppedImage'] = imageDictsLabeled[i]['image'][imageDictsLabeled[i]['boxEdges'][2]:imageDictsLabeled[i]['boxEdges'][3], \
                                                                             imageDictsLabeled[i]['boxEdges'][0]:imageDictsLabeled[i]['boxEdges'][1]]
    return imageDictsLabeled

################################ Display #######################################
# Initial import random subsample.
def displayExamples(imageDicts):
    indices = arange(len(imageDicts))
    shuffle(indices)

    fig = plt.figure(0, (15, 12))
    for i in range(12):
        plt.subplot(3,4, i+1)
        plt.imshow(imageDicts[indices[i]]['image'][125:275, 125:275])
        plt.title(imageDicts[indices[i]]['numFingers'])
# Labled dictionary random subsample.
def displayExamplesLabled(imageDictsLabeled):
    indices = arange(len(imageDictsLabeled))
    shuffle(indices)
    fig = figure(0, (15, 12))
    for i in range(12):
        subplot(3,4, i+1)
        image = imageDictsLabeled[indices[i]]['image'].copy()
        fingerMask = np.zeros((400,400))

        for point in imageDictsLabeled[indices[i]]['indexFingerPoints']:
            fingerMask[point[0], point[1]] = 255

        if imageDictsLabeled[indices[i]]['numFingers']>1:
            for point in imageDictsLabeled[indices[i]]['middleFingerPoints']:
                fingerMask[point[0], point[1]] = 255

        if imageDictsLabeled[indices[i]]['numFingers']>2:
            for point in imageDictsLabeled[indices[i]]['ringFingerPoints']:
                fingerMask[point[0], point[1]] = 255

        image[fingerMask==255] = 255
        imshow(image[125:275, 125:275])

        title(imageDictsLabeled[indices[i]]['numFingers'])
# Dictionary with hands bounded random subsample.
def displayExamplesBounded(imageDictsLabeled):
    indices = arange(len(imageDictsLabeled))
    shuffle(indices)

    fig = figure(0, (15, 12))
    for i in range(12):
        imageDict = imageDictsLabeled[indices[i]]
        image = imageDict['image'].copy()
        handMask = np.zeros((400,400))

        for point in imageDict['box']:
            handMask[point[0], point[1]] = 255

        subplot(3,4,i+1)
        image[handMask==255] = 255
        imshow(image[125:275, 125:275])
# Dictionary with hands bounded random subsample.
def displayExamplesSimplified(imageDictsLabeled):
    indices = arange(len(imageDictsLabeled))
    shuffle(indices)

    fig = figure(0, (15, 12))
    for i in range(12):
        imageDict = imageDictsLabeled[indices[i]]
        image = imageDict['image1bit'].copy()
        handMask = np.zeros((400,400))

        for point in imageDict['box']:
            handMask[point[0], point[1]] = 255

        subplot(3,4,i+1)
        image[handMask==255] = 255
        imshow(image[125:275, 125:275])
