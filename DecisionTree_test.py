# Daniil Koshelyuk exercise on AI: Decision trees led by WelchLabs.

################################## Imports #####################################

from DecisionTree_ownDataHandling import  *
from DecisionTree_preprocessing import  *
from DecisionTree_support import  *
from copy import deepcopy

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

# Import preprocessed data.
data = importPickles(1)
examplesIndicies = [7, 30, 46]
# print (len(data))
# print (data[0].keys())
# displayData(data, examplesIndicies)
# displayThesholdData(data, examplesIndicies, 0)
# displayThesholdData(data, examplesIndicies, 1)
imageList = [data[index] for index in examplesIndicies]
examples, answers = extractExamplesFromList(imageList, imageType='image1bit', pixelOffset=4)
# print examples.shape, answers.shape

# Extract 1 pixel area information.
exampleToShow = np.reshape(examples[421,:], (9,9))
# displayOnePixelArea(exampleToShow)
# print(exampleToShow.astype('int'))

# Test with a simple rule.
rule1 = np.array(([[0, 0, 1, 1, 1, 1, 1, 0, 0],
                   [0, 0, 1, 1, 1, 1, 1, 0, 0],
                   [0, 0, 1, 1, 1, 1, 1, 0, 0],
                   [0, 0, 1, 1, 1, 1, 1, 0, 0],
                   [0, 0, 1, 1, 1, 1, 1, 0, 0],
                   [0, 0, 1, 1, 1, 1, 1, 0, 0],
                   [0, 0, 1, 1, 1, 1, 1, 0, 0],
                   [0, 0, 1, 1, 1, 1, 1, 0, 0],
                   [0, 0, 1, 1, 1, 1, 1, 0, 0]]))
# displayOnePixelArea(rule1)
difference = examples - rule1.ravel()
detectedPixels = np.where(~difference.any(axis=1))[0]
# print(detectedPixels)
# displaySimpleRulePerformance(rules=[rule1], examplesIndicies=examplesIndicies, data=data)

# Test with a set of simple rules.
metaRule = np.array(([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0],
                      [0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0],
                      [0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0],
                      [0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0],
                      [0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0],
                      [0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0],
                      [0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0],
                      [0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0],
                      [0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0]]))
rules = []
sampleWidth = 9
for i in range(6):
    for j in range(5):
        rules.append(metaRule[i:i + sampleWidth, j:j + sampleWidth])
# print len(rules)
# displayGeneratedRules(rules, 6, 5)
# displaySimpleRulePerformance(rules=rules, examplesIndicies=examplesIndicies, data=data)
# displayAutodetectedPixels(data=data, examplesIndicies=examplesIndicies)
matchingIndices = np.array([], dtype='int')
for rule in rules:
    difference      = examples - rule.ravel()
    mI              = np.where(~difference.any(axis=1))[0]
    matchingIndices = np.concatenate((matchingIndices, mI))
detection = np.zeros(examples.shape[0])
detection[matchingIndices] = 1
confusionMatrix, accuracy, recall, precision = computeConfusionMatrix(answers, detection, datails=False)
# print confusionMatrix
# displayConfusionMatrix(rules, examplesIndicies, data, examples, answers)

# Additional mechanism to track different classifiers.
results = {}
results['Simple Rules Classification'] = {'dataset indecies':   deepcopy(examplesIndicies),
                                          'rules':              deepcopy(rules),
                                          'detection':          deepcopy(detection),
                                          'confusion matrix':   deepcopy(confusionMatrix),
                                          'accuracy':           deepcopy(accuracy),
                                          'recall':             deepcopy(recall),
                                          'precision':          deepcopy(precision)}

################################### Lesson 4 ###################################

# Check how many unique examples are there.
print len(answers), sum(answers == 1)

# Obvious redundant rule demonstration - all empty.
# redundantRule1 = np.zeros((9, 9))
# differenceRedundant1 = examples - redundantRule1.ravel()
# detectedPixelsRedundant1 = np.where(~differenceRedundant1.any(axis=1))[0]
# print len(detectedPixelsRedundant1)
# displaySimpleRulePerformance(rules=[redundantRule1], examplesIndicies=examplesIndicies, data=data)

# Obvious redundant rule demonstration - all filled.
# redundantRule2 = np.ones((9, 9))
# differenceRedundant2 = examples - redundantRule2.ravel()
# detectedPixelsRedundant2 = np.where(~differenceRedundant2.any(axis=1))[0]
# print len(detectedPixelsRedundant2)
# displaySimpleRulePerformance(rules=[redundantRule2], examplesIndicies=examplesIndicies, data=data)

# Removing reundancies.
uniqueExamples, uniqueIndicies, uniqueCounts = uniqueRowsCount(examples)
uniqueLabels = answers[uniqueIndicies]
sortedIndicies = np.argsort(uniqueCounts)
# displayMostcommonPixelAreas(uniqueExamples, uniqueCounts, uniqueLabels, sortedIndicies, 40)
fingerExamples, uniqueIndicies, uniqueCounts = uniqueRowsCount(examples[answers == 1])
print len(uniqueExamples), len(fingerExamples)

# Working with first Machine Learning Algorithm - all the positive rules from examples.
# displayPositiveRules(fingerExamples)
rules = fingerExamples.copy()
## - for comparison from before:
# print (results['Simple Rules Classification']['confusion matrix'],
#        results['Simple Rules Classification']['accuracy'],
#        results['Simple Rules Classification']['recall'],
#        results['Simple Rules Classification']['precision'])
## - for comparison without classification:
detection = np.zeros(examples.shape[0])
confusionMatrix, accuracy, recall, precision = computeConfusionMatrix(answers, detection, datails=False)
# displayConfusionMatrix([], examplesIndicies, data, examples, answers)
results['No Classification'] = {'dataset indecies':   deepcopy(examplesIndicies),
                                'rules':              deepcopy([]),
                                'detection':          deepcopy(detection),
                                'confusion matrix':   deepcopy(confusionMatrix),
                                'accuracy':           deepcopy(accuracy),
                                'recall':             deepcopy(recall),
                                'precision':          deepcopy(precision)}
# print (results['No Classification']['confusion matrix'],
#        results['No Classification']['accuracy'],
#        results['No Classification']['recall'],
#        results['No Classification']['precision'])
## - now:
matchingIndices = np.array([], dtype='int')
for rule in rules:
    difference      = examples - rule.ravel()
    mI              = np.where(~difference.any(axis=1))[0]
    matchingIndices = np.concatenate((matchingIndices, mI))
detection = np.zeros(examples.shape[0])
detection[matchingIndices] = 1
confusionMatrix, accuracy, recall, precision = computeConfusionMatrix(answers, detection, datails=False)
# displayConfusionMatrix(rules, examplesIndicies, data, examples, answers)
results['Machine Learning Classification 1'] = {'dataset indecies':   deepcopy(examplesIndicies),
                                                'rules':              deepcopy(rules),
                                                'detection':          deepcopy(detection),
                                                'confusion matrix':   deepcopy(confusionMatrix),
                                                'accuracy':           deepcopy(accuracy),
                                                'recall':             deepcopy(recall),
                                                'precision':          deepcopy(precision)}
# print (results['Machine Learning Classification 1']['confusion matrix'],
#        results['Machine Learning Classification 1']['accuracy'],
#        results['Machine Learning Classification 1']['recall'],
#        results['Machine Learning Classification 1']['precision'])

################################### Lesson 5 ###################################

# Test the behaviours on separate dataset.
testingExamplesIndicies = [34, 45]
testingImageList = [data[index] for index in testingExamplesIndicies]
testingExamples, testingAnswers = extractExamplesFromList(imageList, imageType='image1bit', pixelOffset=4)



## - no classification:
detection = np.zeros(examples.shape[0])
confusionMatrix, accuracy, recall, precision = computeConfusionMatrix(testingAnswers, detection, datails=False)
displayConfusionMatrix(results['No Classification']['rules'], testingExamplesIndicies, data, testingExamples, testingAnswers)
print (confusionMatrix, accuracy, recall, precision)

results['No Classification']['testing examples indicies']                   = deepcopy(testingExamplesIndicies)
results['No Classification']['testing examples']                            = deepcopy(testingExamples)
results['No Classification']['testing answers']                             = deepcopy(testingAnswers)
results['No Classification']['testing detection']                           = deepcopy(detection)
results['No Classification']['testing confusion matrix']                    = deepcopy(confusionMatrix)
results['No Classification']['testing accuracy']                            = deepcopy(accuracy)
results['No Classification']['testing recall']                              = deepcopy(recall)
results['No Classification']['testing precision']                           = deepcopy(precision)

## - simple rules classification: #FIXME Make this into a function.
matchingIndices = np.array([], dtype='int')
for rule in results['Simple Rules Classification']['rules']:
    difference      = examples - rule.ravel()
    mI              = np.where(~difference.any(axis=1))[0]
    matchingIndices = np.concatenate((matchingIndices, mI))
detection = np.zeros(examples.shape[0])
detection[matchingIndices] = 1
confusionMatrix, accuracy, recall, precision = computeConfusionMatrix(testingAnswers, detection, datails=False)
displayConfusionMatrix(results['Simple Rules Classification']['rules'], testingExamplesIndicies, data, testingExamples, testingAnswers)
print (confusionMatrix, accuracy, recall, precision)

results['Simple Rules Classification']['testing examples indicies']         = deepcopy(testingExamplesIndicies)
results['Simple Rules Classification']['testing examples']                  = deepcopy(testingExamples)
results['Simple Rules Classification']['testing answers']                   = deepcopy(testingAnswers)
results['Simple Rules Classification']['testing detection']                 = deepcopy(detection)
results['Simple Rules Classification']['testing confusion matrix']          = deepcopy(confusionMatrix)
results['Simple Rules Classification']['testing accuracy']                  = deepcopy(accuracy)
results['Simple Rules Classification']['testing recall']                    = deepcopy(recall)
results['Simple Rules Classification']['testing precision']                 = deepcopy(precision)

## - Machine Learning classification 1:
matchingIndices = np.array([], dtype='int')
for rule in results['Machine Learning Classification 1']['rules']:
    difference      = examples - rule.ravel()
    mI              = np.where(~difference.any(axis=1))[0]
    matchingIndices = np.concatenate((matchingIndices, mI))
detection = np.zeros(examples.shape[0])
detection[matchingIndices] = 1
confusionMatrix, accuracy, recall, precision = computeConfusionMatrix(testingAnswers, detection, datails=False)
displayConfusionMatrix(results['Machine Learning Classification 1']['rules'], testingExamplesIndicies, data, testingExamples, testingAnswers)
print (confusionMatrix, accuracy, recall, precision)

results['Machine Learning Classification 1']['testing examples indicies']   = deepcopy(testingExamplesIndicies)
results['Machine Learning Classification 1']['testing examples']            = deepcopy(testingExamples)
results['Machine Learning Classification 1']['testing answers']             = deepcopy(testingAnswers)
results['Machine Learning Classification 1']['testing detection']           = deepcopy(detection)
results['Machine Learning Classification 1']['testing confusion matrix']    = deepcopy(confusionMatrix)
results['Machine Learning Classification 1']['testing accuracy']            = deepcopy(accuracy)
results['Machine Learning Classification 1']['testing recall']              = deepcopy(recall)
results['Machine Learning Classification 1']['testing precision']           = deepcopy(precision)
