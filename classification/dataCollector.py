import random
import time

import numpy as np

import mostFrequent
import perceptron
import samples
import sys
import util
import neuralNetwork

TEST_SET_SIZE = 100
DIGIT_DATUM_WIDTH=28
DIGIT_DATUM_HEIGHT=28
FACE_DATUM_WIDTH=60
FACE_DATUM_HEIGHT=70


def basicFeatureExtractorDigit(datum):
    """
    Returns a set of pixel features indicating whether
    each pixel in the provided datum is white (0) or gray/black (1)
    """
    a = datum.getPixels()

    features = util.Counter()
    for x in range(DIGIT_DATUM_WIDTH):
        for y in range(DIGIT_DATUM_HEIGHT):
            if datum.getPixel(x, y) > 0:
                features[(x,y)] = 1
            else:
                features[(x,y)] = 0
    return features

def basicFeatureExtractorFace(datum):
    """
    Returns a set of pixel features indicating whether
    each pixel in the provided datum is an edge (1) or no edge (0)
    """
    a = datum.getPixels()

    features = util.Counter()
    for x in range(FACE_DATUM_WIDTH):
        for y in range(FACE_DATUM_HEIGHT):
            if datum.getPixel(x, y) > 0:
                features[(x,y)] = 1
            else:
                features[(x,y)] = 0
    return features
if __name__ == '__main__':
    for dataType in range(1, 2):
        if dataType == 1:
            print "Data Type: Faces"
            featureFunction = basicFeatureExtractorFace
            rawTrainingData = samples.loadDataFile("facedata/facedatatrain", 451,FACE_DATUM_WIDTH,FACE_DATUM_HEIGHT)
            trainingLabels = samples.loadLabelsFile("facedata/facedatatrainlabels", 451)
            rawTestData = samples.loadDataFile("facedata/facedatatest", 150,FACE_DATUM_WIDTH,FACE_DATUM_HEIGHT)
            testLabels = samples.loadLabelsFile("facedata/facedatatestlabels", 150)
            legalLabels = range(2)
        else:
            print "Data Type: Digits"
            featureFunction = basicFeatureExtractorDigit
            rawTrainingData = samples.loadDataFile("digitdata/trainingimages", 5000,DIGIT_DATUM_WIDTH,DIGIT_DATUM_HEIGHT)
            trainingLabels = samples.loadLabelsFile("digitdata/traininglabels", 5000)
            rawTestData = samples.loadDataFile("digitdata/testimages", 1000,DIGIT_DATUM_WIDTH,DIGIT_DATUM_HEIGHT)
            testLabels = samples.loadLabelsFile("digitdata/testlabels", 1000)
            legalLabels = range(10)
        trainingData = map(featureFunction, rawTrainingData)
        testData = map(featureFunction, rawTestData)
        for algorithm in range(0, 2):
            if algorithm == 0:
                print "Perceptron"
            else:
                print "Two-Layer Neural Network"
            print "Accuracies:"
            for n in range(1, 11):
                accuracies = [float(0)] * 5
                times = [float(0)] * 5
                for i in range(0, 5):
                    if algorithm == 0:
                        iterations = 3
                        classifier = perceptron.PerceptronClassifier(legalLabels, iterations)
                    else:
                        iterations = 1000
                        classifier = neuralNetwork.NeuralNetworkClassifier(legalLabels, iterations)
                    indexes = range(len(trainingData))
                    random.shuffle(indexes)
                    data = [trainingData[0]] * int(n * 0.1 * len(trainingData))
                    labels = [None] * len(data)
                    for x in range(len(data)):
                        data[x] = trainingData[indexes[x]]
                        labels[x] = trainingLabels[indexes[x]]
                    times[i] = time.time()
                    classifier.train(data, labels, None, None)
                    times[i] -= time.time()
                    guesses = classifier.classify(testData)
                    for g in range(len(guesses)):
                        accuracies[i] += guesses[g] == testLabels[g]
                    accuracies[i] /= len(guesses)
                print(str(n*10) + "% of data")
                print "Time:", -1*np.mean(times)
                print "Mean:", np.mean(accuracies)
                print "Standard Deviation:", np.std(accuracies)
