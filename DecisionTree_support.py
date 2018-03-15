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
import cPickle as pickle
from math import ceil, sqrt

################################# General ######################################
# Custom function to help identify finger pixels.
def expandByOne(allPoints, img, relativeThresh=0.5):
    for point in allPoints:
        #There are 8 possible search directions:
        for i in [-1,0,1]:
            for j in [-1,0,1]:
                newPoint = point + [i, j]
                #Make sure point is actually new:
                if (sum(newPoint==point) == 0):
                    if (img[newPoint[0], newPoint[1]] > relativeThresh*img[point[0], point[1]]):
                        allPoints = np.vstack(([allPoints, newPoint]))
    return uniqueRows(allPoints)
# Custom function to identify unique rows in a 2d numpy arrray.
def uniqueRows(a):
    b = np.ascontiguousarray(a).view(np.dtype((np.void, a.dtype.itemsize * a.shape[1])))
    _, i = np.unique(b, return_index=True)
    aUnique = a[i].astype('int')
    return aUnique
# Custom function to identify unique rows in a 2d numpy arrray.
def uniqueRowsCount(a):
    b = np.ascontiguousarray(a).view(np.dtype((np.void, a.dtype.itemsize * a.shape[1])))
    _, i, counts = np.unique(b, return_index=True, return_counts=True)
    aUnique = a[i].astype('int')
    return aUnique, i, counts

################################################################################
########################### Support Functionality ##############################
################################################################################

############################## Pickle Handling #################################
# A function to import data.
def importPickles(type=0):
    if (type==0): # Import raw data.
        #Let's do a list of dicts
        imageDicts = []
        path = '00_data//rawData//'
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
    else: # Import preprocessed data.                                    #TESTED
        path = '00_data//'
        filename = 'fingerDataSet.pickle'
        pickleFileName = path + filename
        pickleFile = open(pickleFileName, 'rb')
        data = pickle.load(pickleFile)
        pickleFile.close()
        return data
# A function to export Labled data.
def exportPickles(imageDictsLabeled, type=0):
    if (type == 0):     # All Data Export
        pickleName = 'fingerDataSet'
        directory = '00_data//'
        pickleFileName = directory + pickleName + ".pickle"
        pickleFile = open(pickleFileName, 'wb')
        pickle.dump(imageDictsLabeled, pickleFile, pickle.HIGHEST_PROTOCOL)
        pickleFile.close()
    elif (type == 1):   # Reduced Data Export
        pickleName = 'fingers'
        directory = '00_data//'
        pickleFileName = directory + pickleName + ".pickle"
        pickleFile = open(pickleFileName, 'wb')
        pickle.dump((fingers, notFingers), pickleFile, pickle.HIGHEST_PROTOCOL)
        pickleFile.close()

############################# Image Analysis ###################################
# Extract a grid of pixel around selected one (default 9 by 9) and
# [finger/not finger] label from imageDict.                              #TESTED
def extractFeatures(imageDict, imageType='image1bit', pixelOffset=4):
    img = imageDict[imageType]
    featuresList = []
    target = np.zeros(imageDict['numPointsInBox'])
    counter = 0
    for i in np.arange(imageDict['boxEdges'][2], imageDict['boxEdges'][3]):
        for j in np.arange(imageDict['boxEdges'][0], imageDict['boxEdges'][1]):
            f = img[i - pixelOffset:i + pixelOffset + 1, j - pixelOffset:j + pixelOffset + 1]
            fVec = f.ravel()
            featuresList.append(fVec)
            #Check and see if this is a finger pixel or not:
            if (np.max(np.sum(imageDict['allFingerPoints'] == [i, j], 1)) == 2):
                target[counter] = 1
            counter += 1
    features = np.vstack((featuresList))
    return features, target
# Extract indivudual examples from list of imageDicts.                   #TESTED
def extractExamplesFromList(imageList, imageType='image1bit', pixelOffset=4):
    allFeaturesList = []
    allTargetList = []
    for i, imageDict in enumerate(imageList):
        features, target = extractFeatures(imageDict, imageType=imageType, pixelOffset=pixelOffset)
        allFeaturesList.append(features)
        allTargetList.append(target)
    examples = np.vstack((allFeaturesList))
    answers = np.hstack((allTargetList))
    return examples, answers
# Convert image to greyscale.                                            #TESTED
def grayscale(imageDict):
    image = np.zeros((imageDict['boxHeight'], imageDict['boxWidth'], 3))
    image[:,:,0] = 1. / 255 * imageDict['image'][imageDict['boxEdges'][2]:imageDict['boxEdges'][3], \
                                                 imageDict['boxEdges'][0]:imageDict['boxEdges'][1]]
    image[:,:,1] = 1. / 255 * imageDict['image'][imageDict['boxEdges'][2]:imageDict['boxEdges'][3], \
                                                 imageDict['boxEdges'][0]:imageDict['boxEdges'][1]]
    image[:,:,2] = 1. / 255 * imageDict['image'][imageDict['boxEdges'][2]:imageDict['boxEdges'][3], \
                                                 imageDict['boxEdges'][0]:imageDict['boxEdges'][1]]
    return image

################################# Colors #######################################
# Return a Linear Segmented Colormap.                                    #TESTED
## sequence: a sequenceuence of floats and RGB-tuples. The floats should be increasing and in the interval (0, 1).
def make_colormap(sequence):
    sequence = [(None,) * 3, 0.0] + list(sequence) + [1.0, (None,) * 3]
    cdict = {'red': [], 'green': [], 'blue': []}
    for i, item in enumerate(sequence):
        if (isinstance(item, float)):
            r1, g1, b1 = sequence[i - 1]
            r2, g2, b2 = sequence[i + 1]
            cdict['red'].append([item, r1, r2])
            cdict['green'].append([item, g1, g2])
            cdict['blue'].append([item, b1, b2])
    return mcolors.LinearSegmentedColormap('CustomMap', cdict)
# Make red and blue colormaps:
c = mcolors.ColorConverter().to_rgb                                      #TESTED
bw  = make_colormap([(1,1,1), (1,1,1), 0.33, c('blue'), c('blue'), 0.66, c('blue')])
rw  = make_colormap([(1,1,1), (1,1,1), 0.33, c('red'), c('red'), 0.66, c('red')])
rwb = make_colormap([c('red'), c('red'), 0.33, (1,1,1), (1,1,1), 0.66, c('blue')])

####################### Decision Tree Functionality ############################
# Compute confusion matrix.                                              #TESTED
def computeConfusionMatrix(answers, detection, datails=True):
    # Make sure data is 1d, not 2d numpy arrays.
    if (answers.ndim == 2):      answers = answers[:,0]
    if (detection.ndim == 2):    detection = detection[:,0]

    TP = np.sum(np.logical_and(answers == 1, detection == 1))
    TN = np.sum(np.logical_and(answers == 0, detection == 0))
    FP = np.sum(np.logical_and(answers == 0, detection == 1))
    FN = np.sum(np.logical_and(answers == 1, detection == 0))
    confusionMatrix = np.array([[TP, FN], [FP, TN]])

    if (TP + FN != 0):  recall = float(TP) / (TP + FN)
    else:               recall = 0
    if (TP + FP != 0):  precision = float(TP) / (TP + FP)
    else:               precision = 0
    accuracy = float(TP + TN) / len(answers)

    if (datails):
        print 'Confusion Matrix:'
        print confusionMatrix
        print 'Recall (TPR) = '     + str(round(recall, 3))     + \
              ' (Portion of fingers that we "caught")'
        print 'Precision (PPV) = '  + str(round(precision, 3))  + \
              '(Portion of predicted finger pixels that were actually finger pixels)'
        print 'Accuracy = '         + str(round(accuracy, 3))

    return confusionMatrix, accuracy, recall, precision
# Test simple rules.                                                     #TESTED
def testSimpleRules(rules, examplesIndicies, data, figure, datails=True):
    for i in range(len(examplesIndicies)):
        subplot             = figure.add_subplot(1, len(examplesIndicies), i + 1)
        imageDict           = data[examplesIndicies[i]]
        examplesFeatures, answersFeatures = extractFeatures(imageDict, imageType='image1bit', pixelOffset=4)
        image               = grayscale(imageDict)
        matchingIndices     = np.array([], dtype='int')
        for rule in rules:
            difference = examplesFeatures - rule.ravel()
            mI = np.where(~difference.any(axis=1))[0]
            matchingIndices = np.concatenate((matchingIndices, mI))
        matchVec = np.zeros(examplesFeatures.shape[0])
        matchVec[matchingIndices] = 1
        matchImage = matchVec.reshape((imageDict['boxHeight'], imageDict['boxWidth']))
        ## Paint with matches:
        image[:,:,0][matchImage==1] = 0
        image[:,:,1][matchImage==1] = 1
        image[:,:,2][matchImage==1] = 0
        subplot.imshow(image, interpolation = 'none')
        subplot.axis('off')
        if (datails):
        	plt.title('Number of Matches = ' + str(sum(matchVec == 1)), fontsize=14)
# # Test logical rules.
# def testLogicalRules(examplesIndicies, data, figure, examples, answers, rule, showLegend=True):
#     for i in range(len(examplesIndicies)):
#         subplot             = figure.add_subplot(1, 4, i + 1)
#         imageDict           = data[examplesIndicies[i]]
#         examplesFeatures, answersFeatures = extractFeatures(imageDict, imageType='image1bit', pixelOffset=4)
#         image               = grayscale(imageDict)
#         yImage              = answersFeatures.reshape(imageDict['boxHeight'], imageDict['boxWidth'])
#
#         # Perform the rule.
#         detection           = rule(examplesFeatures)
#         truePositives       = np.logical_and(answersFeatures, detection)
#         falsePositives      = np.logical_and(np.logical_not(answersFeatures), detection)
#
#         # Colour true positives.
#         ## Paint with matches:
#         image[:,:,0][yImage==1] = 1
#         image[:,:,1][yImage==1] = 1
#         image[:,:,2][yImage==1] = 0
#
#         # Colour true positives.
#         tPImage = truePositives.reshape((imageDict['boxHeight'], imageDict['boxWidth']))
#         ## Paint with matches:
#         image[:,:,0][tPImage==1] = 0
#         image[:,:,1][tPImage==1] = 1
#
#         image[:,:,2][tPImage==1] = 0
#         # Colour false positives.
#         fNImage = falsePositives.reshape((imageDict['boxHeight'], imageDict['boxWidth']))
#         ## Paint with matches:
#         image[:,:,0][fNImage==1] = 1
#         image[:,:,1][fNImage==1] = 0
#         image[:,:,2][fNImage==1] = 0
#
#         # Plot the image.
#         subplot.imshow(image, interpolation = 'none')
#         subplot.axis('off')
#     if (showLegend):
#         legend = Image.open('01_images/legendOne.png', 'r')
#         legend = figure.add_subplot(1, len(examplesIndicies) + 1, len(examplesIndicies) + 1)
#         legend.imshow(legend)
#         legend.axis('off');
#     detection = rule(examples)
#     confusionMatrix, accuracy, recall, precision = computeConfusionMatrix(answers, detection, datails=True)

# Test given rules.                                                      #TESTED
## color: Full, Green. #FIXME Not just texting but drawing...
def testRules(rules, examplesIndicies, data, figure, examples, answers, showLegend=True, color='Full'):
    for i in range(len(examplesIndicies)):
        subplot         = figure.add_subplot(1, 4, i + 1)
        imageDict       = data[examplesIndicies[i]]
        examplesFeatures, answersFeatures = extractFeatures(imageDict, imageType='image1bit', pixelOffset=4)
        image           = grayscale(imageDict)
        matchingIndices = np.array([], dtype='int')
        # Rule type detection.
        if (type(rules) is list):           # List
            for rule in rules:
                diff = examplesFeatures - rule.ravel()
                mI = np.where(~diff.any(axis=1))[0]
                matchingIndices = np.concatenate((matchingIndices, mI))
        elif (type(rules) is np.ndarray):   # Numpy Array
            for i in range(rules.shape[0]):
                diff = examplesFeatures - rules[i, :]
                mI = np.where(~diff.any(axis=1))[0]
                matchingIndices = np.concatenate((matchingIndices, mI))
        elif (callable(rules)):             # Function
            matchingIndices = np.where(rules(examplesFeatures))[0]
        # Colour confusion matrix types on the image.
        matchVec = np.zeros(examplesFeatures.shape[0])
        matchVec[matchingIndices] = 1
        truePositives   = np.logical_and(answersFeatures, matchVec)
        falsePositives  = np.logical_and(np.logical_not(answersFeatures), matchVec)
        falseNegatives  = np.logical_and(answersFeatures, np.logical_not(matchVec))
        # Distinguish selected color scheme.
        if (color == 'Full'):
            # False Negatives.
            fNImage = falseNegatives.reshape(imageDict['boxHeight'], imageDict['boxWidth'])
            ## Paint with matches:
            image[:,:,0][fNImage==1] = 1
            image[:,:,1][fNImage==1] = 1
            image[:,:,2][fNImage==1] = 0

            # True Positives.
            tPImage = truePositives.reshape((imageDict['boxHeight'], imageDict['boxWidth']))
            ## Paint with matches:
            image[:,:,0][tPImage==1] = 0
            image[:,:,1][tPImage==1] = 1
            image[:,:,2][tPImage==1] = 0

            # False Positives.
            fNImage = falsePositives.reshape((imageDict['boxHeight'], imageDict['boxWidth']))
            ## Paint with matches:
            image[:,:,0][fNImage==1] = 1
            image[:,:,1][fNImage==1] = 0
            image[:,:,2][fNImage==1] = 0
        if (color == 'Green'):
            # Matches.
            matchImage = matchVec.reshape((imageDict['boxHeight'], imageDict['boxWidth']))
            ## Paint with matches:
            image[:,:,0][matchImage==1] = 0
            image[:,:,1][matchImage==1] = 1
            image[:,:,2][matchImage==1] = 0
        subplot.imshow(image, interpolation='none')
        subplot.axis('off')
    if (showLegend):
        legend = Image.open('01_images/legendOne.png', 'r')
        legendSubplot = figure.add_subplot(1,4,len(examplesIndicies)+1)
        legendSubplot.imshow(legend)
        legendSubplot.axis('off')

    # Search for matches in all data:
    matchingIndices = np.array([], dtype='int')
    if (type(rules) is list):         # List
        for rule in rules:
            diff = examples - rule.ravel()
            mI = np.where(~diff.any(axis=1))[0]
            matchingIndices = np.concatenate((matchingIndices, mI))
    elif (type(rules) is np.ndarray): # Numpy Array
        for i in range(rules.shape[0]):
            diff = examples - rules[i, :]
            mI = np.where(~diff.any(axis=1))[0]
            matchingIndices = np.concatenate((matchingIndices, mI))
    elif (callable(rules)):           # Function
        matchingIndices = np.where(rules(examples))[0]
    detection = np.zeros(examples.shape[0])
    detection[matchingIndices] = 1
    confusionMatrix, accuracy, recall, precision = computeConfusionMatrix(answers, detection, datails=False)
    return detection, confusionMatrix, accuracy, recall, precision

################################ Display ####################################### #FIXME Clense displays from excersises
# Initial specific cropped image display.                                #TESTED
def displayData(data, examplesIndicies=[1]):
    figure = plt.figure(0, (12, 8))
    for i, exampleIndex in enumerate(examplesIndicies):
        plt.subplot(1, len(examplesIndicies), i + 1)
        plt.imshow(data[exampleIndex]['croppedImage'], cmap='gray', interpolation='none')
        plt.axis('off')
    plt.show()
# Initial specific cropped image thresholded display.                    #TESTED
def displayThesholdData(data, examplesIndicies=[1], type=1):
    figure = plt.figure(0, (12, 8))
    for i, exampleIndex in enumerate(examplesIndicies):
        plt.subplot(1, len(examplesIndicies), i + 1)
        if (type == 1): # Load preprocessed thersholds.
            image = data[exampleIndex]['image1bit'][data[exampleIndex]['boxEdges'][2]:data[exampleIndex]['boxEdges'][3], \
                                                    data[exampleIndex]['boxEdges'][0]:data[exampleIndex]['boxEdges'][1]]
        else: # Direct thresholding.
            image = data[exampleIndex]['croppedImage'] > 92
        plt.imshow(image, cmap='gray', interpolation='none')
        plt.axis('off')
    plt.show()
# Specific pixel area display.                                           #TESTED
def displayOnePixelArea(pixelArea):
    figure = plt.figure(0, (4, 4))
    plt.pcolor(np.flipud(pixelArea), cmap='Greys',  linewidth=.5, color='k', vmin=0, vmax=1)
    plt.show()
# Display performance of a simple rule.                                  #TESTED
def displaySimpleRulePerformance(rules, examplesIndicies, data):
    figure = plt.figure(0, (12, 6))
    testSimpleRules(rules=rules, examplesIndicies=examplesIndicies, data=data, figure=figure)
    plt.show()
# Display a set of rules.                                                #TESTED
def displayGeneratedRules(rules, nr, nc):
    figure = plt.figure(0, (10, 12))
    for i, rule in enumerate(rules):
        plt.subplot(nr, nc, i + 1)
        plt.pcolor(np.flipud(rule), cmap = 'Greys', linewidth=.5, color='k', vmin=0, vmax=1)
        plt.axis('off')
    plt.show()
# Display a autodetected set of Finger Pixels.                           #TESTED
def displayAutodetectedPixels(data, examplesIndicies):
    figure = plt.figure(0, (12, 6))
    for i in range(len(examplesIndicies)):
        figure.add_subplot(1, len(examplesIndicies), i + 1)
        imageDict           = data[examplesIndicies[i]]
        examplesFeatures, answersFeatures = extractFeatures(imageDict, imageType='image1bit', pixelOffset=4)
        image               = grayscale(imageDict)
        yImage              = answersFeatures.reshape(imageDict['boxHeight'], imageDict['boxWidth'])

        # Colour true answers.
        ## Paint with matches:
        image[:,:,0][yImage == 1] = 0
        image[:,:,1][yImage == 1] = .5
        image[:,:,2][yImage == 1] = 1

        # Plot the image.
        plt.imshow(image)
        plt.title('Number of finger pixels = ' + str(sum(answersFeatures == 1)), fontsize=14)
    plt.show()
# Display a multicolor plot of confusion matrix.                         #TESTED
def displayConfusionMatrix(rules, examplesIndicies, data, examples, answers):
    figure = plt.figure(0, (14, 8))
    testRules(rules, examplesIndicies, data, figure, examples, answers)
    plt.show()
# Display most common Pixel areas.                                       #TESTED
def displayMostcommonPixelAreas(uniqueExamples, uniqueCounts, uniqueLabels, sortedIndicies, amount=40):
    figure = plt.figure(0, (16, 12))
    for i in range(amount):
        pixelAreaIndex  = sortedIndicies[-1 * (i + 1)]
        pixelArea       = uniqueExamples[pixelAreaIndex, :].reshape(9,9)
        # Check if a true or false area.
        if (uniqueLabels[pixelAreaIndex]): # True
            colorMap    = bw
            maincolor   = 'b'
        else: # False
            colorMap    = rw
            maincolor   = 'r'
        plt.subplot(ceil(float(amount) / 8), 8, i + 1)
        plt.pcolor(np.flipud(pixelArea), cmap=colorMap,  linewidth=.5, color=maincolor, vmin=0, vmax=1)
        plt.axis('off')
        plt.title(str(uniqueCounts[pixelAreaIndex]))
    plt.show()
# Display all positive Example Pixel Areas. NB! Long                     #TESTED
def displayPositiveRules(fingerExamples):
    figure    = plt.figure(0, (16, 16))
    colorMap  = bw
    maincolor = 'b'
    nr        = ceil(sqrt(float(len(fingerExamples))))
    nc        = ceil(len(fingerExamples) / nr)
    for i in range(len(fingerExamples)):
        rule = fingerExamples[i, :].reshape(9, 9)
        plt.subplot(nr, nc, i + 1)
        plt.pcolor(np.flipud(rule), cmap=colorMap,  linewidth=.5, color=maincolor, vmin=0, vmax=1)
        plt.axis('off')
    plt.show()
