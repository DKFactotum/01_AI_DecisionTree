# Daniil Koshelyuk exercise on AI: Decision trees led by WelchLabs.

################################## Imports #####################################

currentAddress = "D://DAN//00_Documents//03_Programming//AI//WelchLabs//01_AI_DecisionTree//"
import sys, os
sys.path.append(currentAddress + "LeapSDK//lib//x64")
sys.path.append(currentAddress + "LeapSDK//lib")

import Leap as leap
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from time import sleep
from datetime import datetime as dt
from copy import deepcopy

############################# General Functions ################################

def remap(value, oldRange0, oldRange1, newRange0, newRange1):
    return (value - oldRange0) * (newRange1 - newRange0) / (oldRange1 - oldRange0)

################################################################################
########################## Leap Motion Interaction #############################
################################################################################

################################ Data Capture ##################################
# Take a set of frames from leap Motion.
def takeFrames(nShots=3, nCountDown=2):
    lc = leap.Controller()
    lc.set_policy(leap.Controller.POLICY_IMAGES)
    frames = []
    tStart = dt.now()
    for i in range(nShots):
        print "Ready to take a shot", str(i+1)
        for j in reversed(range(nCountDown)):
            print(j+1)
            sleep(1)
        if(lc.is_connected): #controller is a Leap.Controller object
            print "Flash!"
            frames.append(lc.frame())
            print "Done\n"
    print "Data capture done in", (dt.now() - tStart).total_seconds() - nShots * nCountDown, "seconds"
    return frames
# Arrange left camera images from frames into a numpy array.
def arrangeToNumpy(frames):
    tStart = dt.now()
    image_convert = []
    for j in range(len(frames)):
        image = frames[j].images
        image = image[0]
        # print(image.is_valid, image.width, image.height, image.bytes_per_pixel)
        image_data = []
        for i in range(image.width * image.height):
            image_data.append(image.data[i])
        image_data = np.array(image_data, dtype=float)
        image_data = np.reshape(image_data, (image.height, image.width))
        image_data = np.flip(image_data, axis=1)
        image_data = image_data/np.amax(image_data)
        image_convert.append(image_data)
    print "Data conversion done in", dt.now() - tStart, "seconds"
    return image_convert
# Save Numpy Images
def storeNumpyText(image_convert):
    path = "Pictures//"
    i = 0
    for image in image_convert:
        np.savetxt(path + "frame" + str(i) + ".data", image)
        i += 1

################################################################################
############################### Image Analysis #################################
################################################################################

############################## Database Reading ################################
# Read Captured oimages data.
def read():
    path = "Pictures//"
    filenames = []
    for filename in os.listdir(path):
        if filename.endswith(".data"):
            filenames.append(filename)
    images = []
    for filename in filenames:
        print (filename)
        try: # Catch mistakes in formats.
            images.append(np.loadtxt(path + filename))
        except:
            pass
    return images

############################## Image Processing ################################
# Crop image from (xo,yo) to (x1,y1) pixels.
def crop(image, x0, y0, x1, y1):
    if (x0 < 0):
        x0 = 0
    if (y0 < 0):
        y0 = 0
    if (x1 > image.shape[1]):
        x1 = image.shape[1]
    if (y1 > image.shape[0]):
        y1 = image.shape[0]
    return image[y0:y1, x0:x1]
# Crop an image with width and height parameters
def cropSize(image, x0, y0, w=100, h=100):
    return crop(image, x0, y0, x0 + w, y0 + h)
# Scale image avereging information.
def scale(image, factor):
    image_scaled = []
    oldSizeX = image.shape[1]
    oldSizeY = image.shape[0]
    newSizeX = int(oldSizeX * factor)
    newSizeY = int(oldSizeY * factor)
    # print(oldSizeX, oldSizeY, newSizeX, newSizeY)
    for i in range(0, newSizeY):
        y = int(remap(i, 0, newSizeY, 0, oldSizeY))
        y1 = int(remap(i + 1, 0, newSizeY, 0, oldSizeY))
        for j in range(0, newSizeX):
            x = int(remap(j, 0, newSizeX, 0, oldSizeX))
            x1 = int(remap(j + 1, 0, newSizeX, 0, oldSizeX))
            newPixel = np.mean(image[y:y1, x:x1])
            image_scaled.append(newPixel)
    image_scaled = np.array(image_scaled)
    image_scaled = np.reshape(image_scaled, (newSizeY, newSizeX))
    return image_scaled
# 1 bit Image Thresholding.
def thershold(image, level):
    value = remap(level, 0.0, 1.0, float(np.amin(image)), float(np.amax(image)))
    image_thershold = (image > value)
    return np.array(image_thershold, dtype=float)
# Extract area around a pixel.
def getArea(image, xindex, yindex, size=4):
    if (xindex <= size):
        xindex = size
    if (yindex <= size):
        yindex = size
    if (xindex > image.shape[1] - size):
        xindex = image.shape[1] - size
    if (yindex > image.shape[0] - size):
        yindex = image.shape[0] - size
    else:
        y0 = yindex - size
        y1 = yindex + size
        x0 = xindex - size
        x1 = xindex + size
        area = image[y0:y1, x0:x1]
        mask = deepcopy(image)
        # mask = np.zeros(image.shape, dtype=float)
        mask[y0:y1, x0:x1] = .5
        return area, mask

################################ Display #######################################
# Display Import Data
def displayImages(images):
    plt.figure(1)
    n = len(images)
    for i, image in enumerate(images):
        plt.subplot(n,1,i+1)
        plt.imshow(images[i])
    plt.show()
# Display
def displayImageProcessingSteps(images, nr=2, nc=3):
    plt.figure(1)
    for i, image in enumerate(images):
        plt.subplot(nr,nc,i+1)
        plt.imshow(image[0], cmap="gray", interpolation="none")
        plt.title(image[1])
        # x0 = pixels[0][0] - sampleSize - 1
        # y0 = pixels[0][1] - sampleSize - 1
        # w = area.shape[1] + 1
        # h = area.shape[0] + 1
        # print(x0, y0, w, h)
        # patches.Rectangle((x0, y0), w, h, fill=False, edgecolor="red")
    plt.show()
