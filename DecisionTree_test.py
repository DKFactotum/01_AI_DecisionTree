# Daniil Koshelyuk exercise on AI: Decision trees led by WelchLabs.

################################## Imports #####################################

from DecisionTree_ownDataHandling import  *
from DecisionTree_preprocessing import  *
from DecisionTree_support import  *

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
    images              = read()
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
    imageDicts = importPickles()
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
# doCapture()
doAnalyze()

# doDataPrePorcessing()
