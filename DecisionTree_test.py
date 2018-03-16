# Daniil Koshelyuk exercise on AI: Decision trees led by WelchLabs.

################################## Imports #####################################

from DecisionTree_ownDataHandling import  *
from DecisionTree_preprocessing import  *
from DecisionTree_support import  *
from copy import deepcopy
from time import time
from math import floor
from itertools import combinations

################################################################################
############################## Own Experiments #################################
################################################################################

# Go through all Data Capture Steps.
def doCapture():
    frames = takeFrames()
    image_convert = arrangeToNumpy(frames)
    displayImages(image_convert)
    storeNumpyText(image_convert)
# Go through all the analysis steps.
def doAnalyze():
    images              = readNumpyText()
    # displayImages(images)
    image               = images[0]
    image_threshold     = thershold(image, .63)
    image_crop          = cropSize(image_threshold, 250, 40, 200, 200)
    image_scale         = scale(image_crop, .5)
    image_scale         = thershold(image_scale, .4) # to clean the avereging on edges
    pixels = [[45,36]]
    sampleSize = 5
    area, image_mask    = getArea(image_scale, pixels[0][0], pixels[0][1], sampleSize)

    displayImages = [[image, "Original"],
                     [image_threshold, "Thresholded"],
                     [image_crop, "Cropped"],
                     [image_scale, "Scaled down"],
                     [area, "Pixel Selection"],
                     [image_mask, "Pixel Selection Highlight"]]
    displayImageProcessingSteps(displayImages)

################################################################################
############################ Data Preprocessing ################################
################################################################################

# Go through all the preprocessing steps.
def doDataPrePorcessing():
    imageDicts = importPickles(0)
    displayExamples(imageDicts)
    imageDictsLabeled = labelImageDicts(imageDicts)
    displayExamplesLabled(imageDictsLabeled)
    imageDictsLabeled = coverHand(imageDictsLabeled) # Reference or object? Do I need to pass it back?
    imageDictsLabeled = boundingBox(imageDictsLabeled)
    displayExamplesBounded(imageDictsLabeled)
    imageDictsLabeled = reduceBits(imageDictsLabeled)
    displayExamplesSimplified(imageDictsLabeled)
    imageDictsLabeled = arrangeCroppedImages(imageDictsLabeled)
    exportPickles(imageDictsLabeled)

################################################################################
################################ Active Zone ###################################
################################################################################
# Own tries.
# # doCapture()
# doAnalyze()

# Data Capture. NB! Check
# doDataPrePorcessing()

# Following the course:
################################## Lesson 1-3 ##################################

timeStart = time()
timeStartTotal = timeStart

# Import preprocessed data.
data = importPickles(1)
# examplesIndicies = [7, 30, 46]
# # print (len(data))
# # print (data[0].keys())
# # displayData(data, examplesIndicies)
# # displayThesholdData(data, examplesIndicies, 0)
# # displayThesholdData(data, examplesIndicies, 1)
# imageList = [data[index] for index in examplesIndicies]
# examples, answers = extractExamplesFromList(imageList, imageType='image1bit', pixelOffset=4)
# # print examples.shape, answers.shape
#
# # Extract 1 pixel area information.
# exampleToShow = np.reshape(examples[421,:], (9,9))
# # displayOnePixelArea(exampleToShow)
# # print(exampleToShow.astype('int'))
#
# # Test with a simple rule.
# rule1 = np.array(([[0, 0, 1, 1, 1, 1, 1, 0, 0],
#                    [0, 0, 1, 1, 1, 1, 1, 0, 0],
#                    [0, 0, 1, 1, 1, 1, 1, 0, 0],
#                    [0, 0, 1, 1, 1, 1, 1, 0, 0],
#                    [0, 0, 1, 1, 1, 1, 1, 0, 0],
#                    [0, 0, 1, 1, 1, 1, 1, 0, 0],
#                    [0, 0, 1, 1, 1, 1, 1, 0, 0],
#                    [0, 0, 1, 1, 1, 1, 1, 0, 0],
#                    [0, 0, 1, 1, 1, 1, 1, 0, 0]]))
# # displayOnePixelArea(rule1)
# difference = examples - rule1.ravel()
# detectedPixels = np.where(~difference.any(axis=1))[0]
# # print(detectedPixels)
# # displaySimpleRulePerformance(rules=[rule1], examplesIndicies=examplesIndicies, data=data)
#
# # Test with a set of simple rules.
# metaRule = np.array(([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#                       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#                       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#                       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#                       [0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0],
#                       [0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0],
#                       [0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0],
#                       [0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0],
#                       [0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0],
#                       [0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0],
#                       [0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0],
#                       [0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0],
#                       [0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0],
#                       [0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0]]))
# rules = []
# sampleWidth = 9
# for i in range(6):
#     for j in range(5):
#         rules.append(metaRule[i:i + sampleWidth, j:j + sampleWidth])
# # print len(rules)
# # displayGeneratedRules(rules, 6, 5)
# # displaySimpleRulePerformance(rules=rules, examplesIndicies=examplesIndicies, data=data)
# # displayAutodetectedPixels(data=data, examplesIndicies=examplesIndicies)
# matchingIndices = np.array([], dtype='int')
# for rule in rules:
#     difference      = examples - rule.ravel()
#     mI              = np.where(~difference.any(axis=1))[0]
#     matchingIndices = np.concatenate((matchingIndices, mI))
# detection = np.zeros(examples.shape[0])
# detection[matchingIndices] = 1
# confusionMatrix, accuracy, recall, precision = computeConfusionMatrix(answers, detection, details=False)
# # print confusionMatrix
# # displayConfusionMatrix(rules, examplesIndicies, data, examples, answers)
#
# # Additional mechanism to track different classifiers.
# results = {}
# results['Simple Rules Classification'] = {'dataset indecies':   deepcopy(examplesIndicies),
#                                           'rules':              deepcopy(rules),
#                                           'detection':          deepcopy(detection),
#                                           'confusion matrix':   deepcopy(confusionMatrix),
#                                           'accuracy':           deepcopy(accuracy),
#                                           'recall':             deepcopy(recall),
#                                           'precision':          deepcopy(precision)}
#
# print "Time up till Lesson 4 " + str(round(time() - timeStart, 3)) + ", total of " + str(round(time() - timeStartTotal, 3))
# timeStart = time()
#
# ################################### Lesson 4 ###################################
#
# # Check how many unique examples are there.
# print len(answers), sum(answers == 1)
#
# # Obvious redundant rule demonstration - all empty.
# # redundantRule1 = np.zeros((9, 9))
# # differenceRedundant1 = examples - redundantRule1.ravel()
# # detectedPixelsRedundant1 = np.where(~differenceRedundant1.any(axis=1))[0]
# # print len(detectedPixelsRedundant1)
# # displaySimpleRulePerformance(rules=[redundantRule1], examplesIndicies=examplesIndicies, data=data)
#
# # Obvious redundant rule demonstration - all filled.
# # redundantRule2 = np.ones((9, 9))
# # differenceRedundant2 = examples - redundantRule2.ravel()
# # detectedPixelsRedundant2 = np.where(~differenceRedundant2.any(axis=1))[0]
# # print len(detectedPixelsRedundant2)
# # displaySimpleRulePerformance(rules=[redundantRule2], examplesIndicies=examplesIndicies, data=data)
#
# # Removing reundancies.
# # uniqueExamples, uniqueIndicies, uniqueCounts = uniqueRowsCount(examples)
# # uniqueLabels = answers[uniqueIndicies]
# # sortedIndicies = np.argsort(uniqueCounts)
# # displayMostcommonPixelAreas(uniqueExamples, uniqueCounts, uniqueLabels, sortedIndicies, 40)
# fingerExamples, uniqueIndicies, uniqueCounts = uniqueRowsCount(examples[answers == 1])
# # print len(uniqueExamples), len(fingerExamples)
#
# # Working with first Machine Learning Algorithm - all the positive rules from examples.
# # displayPositiveRules(fingerExamples)
# rules = fingerExamples.copy()
#
# ## - for comparison without classification:
# detection = np.zeros(examples.shape[0])
# confusionMatrix, accuracy, recall, precision = computeConfusionMatrix(answers, detection, details=False)
# # displayConfusionMatrix([], examplesIndicies, data, examples, answers)
# # print (confusionMatrix, accuracy, recall, precision)
# results['No Classification'] = {'dataset indecies':   deepcopy(examplesIndicies),
#                                 'rules':              deepcopy([]),
#                                 'detection':          deepcopy(detection),
#                                 'confusion matrix':   deepcopy(confusionMatrix),
#                                 'accuracy':           deepcopy(accuracy),
#                                 'recall':             deepcopy(recall),
#                                 'precision':          deepcopy(precision)}
# ## - for comparison from before:
# # print (results['Simple Rules Classification']['confusion matrix'],
# #        results['Simple Rules Classification']['accuracy'],
# #        results['Simple Rules Classification']['recall'],
# #        results['Simple Rules Classification']['precision'])
#
# ## - time:
# matchingIndices = np.array([], dtype='int')
# for rule in rules:
#     difference      = examples - rule.ravel()
#     mI              = np.where(~difference.any(axis=1))[0]
#     matchingIndices = np.concatenate((matchingIndices, mI))
# detection = np.zeros(examples.shape[0])
# detection[matchingIndices] = 1
# confusionMatrix, accuracy, recall, precision = computeConfusionMatrix(answers, detection, details=False)
# # displayConfusionMatrix(rules, examplesIndicies, data, examples, answers)
# # print (confusionMatrix, accuracy, recall, precision)
# results['Machine Learning Classification 1'] = {'dataset indecies':   deepcopy(examplesIndicies),
#                                                 'rules':              deepcopy(rules),
#                                                 'detection':          deepcopy(detection),
#                                                 'confusion matrix':   deepcopy(confusionMatrix),
#                                                 'accuracy':           deepcopy(accuracy),
#                                                 'recall':             deepcopy(recall),
#                                                 'precision':          deepcopy(precision)}
#
# print "Time up till Lesson 5 " + str(round(time() - timeStart, 3)) + ", total of " + str(round(time() - timeStartTotal, 3))
# timeStart = time()
#
# ################################### Lesson 5 ###################################
#
# # Test the behaviours on separate dataset.
# testingExamplesIndicies = [34, 45]
# testingImageList = [data[index] for index in testingExamplesIndicies]
# testingExamples, testingAnswers = extractExamplesFromList(testingImageList, imageType='image1bit', pixelOffset=4)
#
# ## - no classification:
# detection = np.zeros(testingExamples.shape[0])
# confusionMatrix, accuracy, recall, precision = computeConfusionMatrix(testingAnswers, detection, details=False)
# # displayConfusionMatrix(results['No Classification']['rules'], testingExamplesIndicies, data, testingExamples, testingAnswers)
# # print (confusionMatrix, accuracy, recall, precision)
#
# results['No Classification']['testing examples indicies']                   = deepcopy(testingExamplesIndicies)
# results['No Classification']['testing examples']                            = deepcopy(testingExamples)
# results['No Classification']['testing answers']                             = deepcopy(testingAnswers)
# results['No Classification']['testing detection']                           = deepcopy(detection)
# results['No Classification']['testing confusion matrix']                    = deepcopy(confusionMatrix)
# results['No Classification']['testing accuracy']                            = deepcopy(accuracy)
# results['No Classification']['testing recall']                              = deepcopy(recall)
# results['No Classification']['testing precision']                           = deepcopy(precision)
#
# ## - simple rules classification: #FIXME Make this into a function.
# matchingIndices = np.array([], dtype='int')
# for rule in results['Simple Rules Classification']['rules']:
#     difference      = testingExamples - rule.ravel()
#     mI              = np.where(~difference.any(axis=1))[0]
#     matchingIndices = np.concatenate((matchingIndices, mI))
# detection = np.zeros(testingExamples.shape[0])
# detection[matchingIndices] = 1
# confusionMatrix, accuracy, recall, precision = computeConfusionMatrix(testingAnswers, detection, details=False)
# # displayConfusionMatrix(results['Simple Rules Classification']['rules'], testingExamplesIndicies, data, testingExamples, testingAnswers)
# # print (confusionMatrix, accuracy, recall, precision)
#
# results['Simple Rules Classification']['testing examples indicies']         = deepcopy(testingExamplesIndicies)
# results['Simple Rules Classification']['testing examples']                  = deepcopy(testingExamples)
# results['Simple Rules Classification']['testing answers']                   = deepcopy(testingAnswers)
# results['Simple Rules Classification']['testing detection']                 = deepcopy(detection)
# results['Simple Rules Classification']['testing confusion matrix']          = deepcopy(confusionMatrix)
# results['Simple Rules Classification']['testing accuracy']                  = deepcopy(accuracy)
# results['Simple Rules Classification']['testing recall']                    = deepcopy(recall)
# results['Simple Rules Classification']['testing precision']                 = deepcopy(precision)
#
# ## - Machine Learning classification 1:
# matchingIndices = np.array([], dtype='int')
# for rule in results['Machine Learning Classification 1']['rules']:
#     difference      = testingExamples - rule.ravel()
#     mI              = np.where(~difference.any(axis=1))[0]
#     matchingIndices = np.concatenate((matchingIndices, mI))
# detection = np.zeros(testingExamples.shape[0])
# detection[matchingIndices] = 1
# confusionMatrix, accuracy, recall, precision = computeConfusionMatrix(testingAnswers, detection, details=False)
# # displayConfusionMatrix(results['Machine Learning Classification 1']['rules'], testingExamplesIndicies, data, testingExamples, testingAnswers)
# # print (confusionMatrix, accuracy, recall, precision)
#
# results['Machine Learning Classification 1']['testing examples indicies']   = deepcopy(testingExamplesIndicies)
# results['Machine Learning Classification 1']['testing examples']            = deepcopy(testingExamples)
# results['Machine Learning Classification 1']['testing answers']             = deepcopy(testingAnswers)
# results['Machine Learning Classification 1']['testing detection']           = deepcopy(detection)
# results['Machine Learning Classification 1']['testing confusion matrix']    = deepcopy(confusionMatrix)
# results['Machine Learning Classification 1']['testing accuracy']            = deepcopy(accuracy)
# results['Machine Learning Classification 1']['testing recall']              = deepcopy(recall)
# results['Machine Learning Classification 1']['testing precision']           = deepcopy(precision)
#
#
# # To address the low performance on testing data set will try to train on more examples.
# ## New rules:
# examplesIndicies = range(30)
# imageList = [data[index] for index in examplesIndicies]
# examples, answers = extractExamplesFromList(imageList, imageType='image1bit', pixelOffset=4)
# fingerExamples, uniqueIndicies, uniqueCounts = uniqueRowsCount(examples[answers == 1])
# rules = fingerExamples.copy()
#
# ## Training:
# matchingIndices = np.array([], dtype='int')
# for rule in rules:
#     difference      = examples - rule.ravel()
#     mI              = np.where(~difference.any(axis=1))[0]
#     matchingIndices = np.concatenate((matchingIndices, mI))
# detection = np.zeros(examples.shape[0])
# detection[matchingIndices] = 1
# confusionMatrix, accuracy, recall, precision = computeConfusionMatrix(answers, detection, details=False)
# # displayConfusionMatrix(rules, examplesIndicies, data, examples, answers)
# # print (confusionMatrix, accuracy, recall, precision)
#
# results['Machine Learning Classification 2'] = {'dataset indecies':   deepcopy(examplesIndicies),
#                                                 'rules':              deepcopy(rules),
#                                                 'detection':          deepcopy(detection),
#                                                 'confusion matrix':   deepcopy(confusionMatrix),
#                                                 'accuracy':           deepcopy(accuracy),
#                                                 'recall':             deepcopy(recall),
#                                                 'precision':          deepcopy(precision)}
# ## Testing:
# matchingIndices = np.array([], dtype='int')
# for rule in results['Machine Learning Classification 1']['rules']:
#     difference      = testingExamples - rule.ravel()
#     mI              = np.where(~difference.any(axis=1))[0]
#     matchingIndices = np.concatenate((matchingIndices, mI))
# detection = np.zeros(testingExamples.shape[0])
# detection[matchingIndices] = 1
# confusionMatrix, accuracy, recall, precision = computeConfusionMatrix(testingAnswers, detection, details=False)
# # displayConfusionMatrix(results['Machine Learning Classification 1']['rules'], testingExamplesIndicies, data, testingExamples, testingAnswers)
# # print (confusionMatrix, accuracy, recall, precision)
#
# results['Machine Learning Classification 2']['testing examples indicies']   = deepcopy(testingExamplesIndicies)
# results['Machine Learning Classification 2']['testing examples']            = deepcopy(testingExamples)
# results['Machine Learning Classification 2']['testing answers']             = deepcopy(testingAnswers)
# results['Machine Learning Classification 2']['testing detection']           = deepcopy(detection)
# results['Machine Learning Classification 2']['testing confusion matrix']    = deepcopy(confusionMatrix)
# results['Machine Learning Classification 2']['testing accuracy']            = deepcopy(accuracy)
# results['Machine Learning Classification 2']['testing recall']              = deepcopy(recall)
# results['Machine Learning Classification 2']['testing precision']           = deepcopy(precision)
#
# print "Time up till Lesson 9 " + str(round(time() - timeStart, 3)) + ", total of " + str(round(time() - timeStartTotal, 3))
# timeStart = time()

################################### Lesson 9 ###################################

# Over 5 minutes in calculations up to now.
examplesIndicies = [7, 30, 46]
imageList = [data[index] for index in examplesIndicies]
examples, answers = extractExamplesFromList(imageList, imageType='image1bit', pixelOffset=4)

testingExamplesIndicies = [40, 41]
testingImageList = [data[index] for index in testingExamplesIndicies]
testingExamples, testingAnswers = extractExamplesFromList(testingImageList, imageType='image1bit', pixelOffset=4)

# # Test one step more variant solution than no classification one.
# rules = []
# numErrors = []
# sampleWidth = 9
# timeStartLocal = time()
# for i in range(sampleWidth * sampleWidth):
#     # The rule for the pixel = 0
#     detection = (examples[:,i] == 0)
#     numErrors.append(sum(abs(detection - answers)))
#     rules.append([i, 0])
#     # The rule for the pixel = 1
#     detection = (examples[:,i] == 1)
#     numErrors.append(sum(abs(detection - answers)))
#     rules.append([i, 1])
# print "Time for cycle " + str(round(time() - timeStartLocal, 3))
# ruleIndex = np.argmin(np.array(numErrors))
# # print ruleIndex
# ruleIndex = int(floor(float(ruleIndex)/2))
# rule = np.zeros(sampleWidth * sampleWidth)
# rule[ruleIndex] = 1
# rule = np.reshape(rule, (9, 9))
# # displayOnePixelArea(rule)
# displayLogicalRulePerformance(ruleIndex=ruleIndex, examplesIndicies=examplesIndicies, data=data)
# matchIndices = np.where(examples[:,ruleIndex]==1)
# detection = np.zeros(examples.shape[0])
# detection[matchIndices] = 1
# confusionMatrix, accuracy, recall, precision = computeConfusionMatrix(answers, detection, details=False)
# print (confusionMatrix, accuracy, recall, precision)
#
# displayLogicalRulePerformance(ruleIndex=int(floor(float(ruleIndex)/2)), examplesIndicies=testingExamplesIndicies, data=data)
# matchIndices = np.where(testingExamples[:,int(floor(float(ruleIndex)/2))]==1)
# detection = np.zeros(testingExamples.shape[0])
# detection[matchIndices] = 1
# confusionMatrix, accuracy, recall, precision = computeConfusionMatrix(testingAnswers, detection, details=False)
# print (confusionMatrix, accuracy, recall, precision)
#
# print "Time up till Lesson 10 " + str(round(time() - timeStart, 3)) + ", total of " + str(round(time() - timeStartTotal, 3))
# timeStart = time()

################################## Lesson 10 ###################################

# # Test one step more variant solution than onex pixel logical rules.
# rules = []
# numErrors = []
# sampleWidth = 9
# timeStartLocal = time()
# for i in range(sampleWidth * sampleWidth):
#     for j in range(sampleWidth * sampleWidth):
#         if (i != j):
#             # The rule for the pixel1 = 0, pixel2 = 0
#             detection = np.logical_and((examples[:,i] == 0), (examples[:,j] == 0))
#             numErrors.append(sum(abs(detection - answers)))
#             rules.append([i, j, 0, 0])
#             # The rule for the pixel1 = 1, pixel2 = 0
#             detection = np.logical_and((examples[:,i] == 1), (examples[:,j] == 0))
#             numErrors.append(sum(abs(detection - answers)))
#             rules.append([i, j, 1, 0])
#             # The rule for the pixel1 = 0, pixel2 = 1
#             detection = np.logical_and((examples[:,i] == 0), (examples[:,j] == 1))
#             numErrors.append(sum(abs(detection - answers)))
#             rules.append([i, j, 0, 1])
#             # The rule for the pixel1 = 1, pixel2 = 1
#             detection = np.logical_and((examples[:,i] == 1), (examples[:,j] == 1))
#             numErrors.append(sum(abs(detection - answers)))
#             rules.append([i, j, 1, 1])
# print "Time for cycle " + str(round(time() - timeStartLocal, 3))
# ruleIndex = np.argmin(np.array(numErrors))
# print len(rules), len(numErrors), ruleIndex, rules[ruleIndex]
# displayLogicalRule(rules[ruleIndex])
#
# rule = lambda X: np.logical_and(X[:, rules[ruleIndex][0]] == rules[ruleIndex][2], X[:,rules[ruleIndex][1]] == rules[ruleIndex][3])
# displayLogicalExecutableRulePerformance(rule, examplesIndicies, data, examples, answers)
# displayLogicalExecutableRulePerformance(rule, testingExamplesIndicies, data, testingExamples, testingAnswers)

# Test one step more variant solution than two pixel logical rules.
rules = []
numErrors = []
sampleWidth = 9
timeStartLocal = time()
for c in combinations(range(sampleWidth * sampleWidth), 3):
    i = c[0]
    j = c[1]
    k = c[2]
    # The rule for the pixel1 = 0, pixel2 = 0, pixel3 = 0
    detection = np.logical_and(np.logical_and((examples[:,i] == 0), (examples[:,j] == 0)), (examples[:,k] == 0))
    numErrors.append(sum(abs(detection - answers)))
    rules.append([i, j, k, 0, 0, 0])
    # The rule for the pixel1 = 0, pixel2 = 0, pixel3 = 1
    detection = np.logical_and(np.logical_and((examples[:,i] == 0), (examples[:,j] == 0)), (examples[:,k] == 1))
    numErrors.append(sum(abs(detection - answers)))
    rules.append([i, j, k, 0, 0, 1])
    # The rule for the pixel1 = 0, pixel2 = 1, pixel3 = 0
    detection = np.logical_and(np.logical_and((examples[:,i] == 0), (examples[:,j] == 1)), (examples[:,k] == 0))
    numErrors.append(sum(abs(detection - answers)))
    rules.append([i, j, k, 0, 1, 0])
    # The rule for the pixel1 = 0, pixel2 = 1, pixel3 = 1
    detection = np.logical_and(np.logical_and((examples[:,i] == 0), (examples[:,j] == 1)), (examples[:,k] == 1))
    numErrors.append(sum(abs(detection - answers)))
    rules.append([i, j, k, 0, 1, 1])
    # The rule for the pixel1 = 1, pixel2 = 0, pixel3 = 0
    detection = np.logical_and(np.logical_and((examples[:,i] == 1), (examples[:,j] == 0)), (examples[:,k] == 0))
    numErrors.append(sum(abs(detection - answers)))
    rules.append([i, j, k, 1, 0, 0])
    # The rule for the pixel1 = 1, pixel2 = 0, pixel3 = 1
    detection = np.logical_and(np.logical_and((examples[:,i] == 1), (examples[:,j] == 0)), (examples[:,k] == 1))
    numErrors.append(sum(abs(detection - answers)))
    rules.append([i, j, k, 1, 0, 1])
    # The rule for the pixel1 = 1, pixel2 = 1, pixel3 = 0
    detection = np.logical_and(np.logical_and((examples[:,i] == 1), (examples[:,j] == 1)), (examples[:,k] == 0))
    numErrors.append(sum(abs(detection - answers)))
    rules.append([i, j, k, 1, 1, 0])
    # The rule for the pixel1 = 1, pixel2 = 1, pixel3 = 1
    detection = np.logical_and(np.logical_and((examples[:,i] == 1), (examples[:,j] == 1)), (examples[:,k] == 1))
    numErrors.append(sum(abs(detection - answers)))
    rules.append([i, j, k, 1, 1, 1])
print "Time for cycle " + str(round(time() - timeStartLocal, 3))
ruleIndex = np.argmin(np.array(numErrors))
print len(rules), len(numErrors), ruleIndex, rules[ruleIndex]
displayLogicalRule(rules[ruleIndex])

rule = lambda X: np.logical_and(np.logical_and(X[:, rules[ruleIndex][0]] == rules[ruleIndex][3], X[:,rules[ruleIndex][1]] == rules[ruleIndex][4]), X[:,rules[ruleIndex][2]] == rules[ruleIndex][5])
displayLogicalExecutableRulePerformance(rule, examplesIndicies, data, examples, answers)
displayLogicalExecutableRulePerformance(rule, testingExamplesIndicies, data, testingExamples, testingAnswers)

print "Time up till the end " + str(round(time() - timeStart, 3)) + ", total of " + str(round(time() - timeStartTotal, 3))
