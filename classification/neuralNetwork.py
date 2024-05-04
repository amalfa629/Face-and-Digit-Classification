# neuralNetwork.py
# -------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html
import random

# Perceptron implementation
import util
import math
import numpy as np
PRINT = True


class NeuralNetworkClassifier:
    """
    Neural Network classifier.

    Note that the variable 'datum' in this code refers to a counter of features
    (not to a raw samples.Datum).
    """
    def __init__( self, legalLabels, max_iterations):
        self.legalLabels = legalLabels
        self.type = "neuralnetwork"
        self.max_iterations = max_iterations
        self.hiddenNodes = 10
        self.weights = {}
        self.lamb = 0.01
        self.alpha = 1

    def train( self, trainingData, trainingLabels, validationData, validationLabels):
        self.features = trainingData[0].keys() # could be useful later
        n = len(trainingData)
        g = np.vectorize(lambda z: 1/float(1+math.exp(-z)))
        for layer in range(0, 2):
            self.weights[layer] = [[0] * ((-1*layer + 1)*len(self.features) + layer*self.hiddenNodes + 1)] * ((-1*layer + 1)*self.hiddenNodes + layer*len(self.legalLabels))
            for i in range(len(self.weights[layer])):
                #self.weights[layer][i] = np.random.uniform(-1/math.sqrt(len(self.features)), 1/math.sqrt(len(self.features)), ((-1*layer + 1)*len(self.features) + layer*self.hiddenNodes + 1))
                self.weights[layer][i] = np.random.uniform(-1, 1, ((-1*layer + 1)*len(self.features) + layer*self.hiddenNodes + 1))
            self.weights[layer] = np.array(self.weights[layer])
        previousAccuracies = np.array([np.nan] * 10)
        for iteration in range(self.max_iterations):
            cost = 0
            if self.alpha <= 0.001:
                return
            print "Starting iteration ", iteration, "..."
            correct = 0
            gradients = [np.zeros_like(self.weights[0]), np.zeros_like(self.weights[1])]
            for image in range(n):
                x = [None] * len(self.features)
                c = 0
                for feature in self.features:
                    x[c] = trainingData[image][feature]
                    c += 1
                x = np.array(x)
                y = np.array([0] * len(self.legalLabels))
                y[trainingLabels[image]] = 1
                a = [x]
                error = {}
                for layer in range(1, 3):
                    a[layer - 1] = np.insert(a[layer - 1], 0, 1)
                    z = np.matmul(self.weights[layer - 1], a[layer - 1])
                    a.append(g(z))
                if np.argmax(a[2]) == np.argmax(y):
                    correct += 1
                error[2] = np.array(a[2] - y)
                error[1] = np.multiply(np.matmul(self.weights[1].T, error[2]), np.multiply(a[1], 1 - a[1]))[1:self.hiddenNodes + 1]
                for layer in range(1, 3):
                    gradients[layer - 1] = gradients[layer - 1] + np.outer(error[layer], a[layer - 1])
                cost += np.sum(y * np.log(a[2]) + (1 - y) * np.log(1 - a[2]))
            cost *= (-1/float(n))
            D = [np.zeros_like(self.weights[0]), np.zeros_like(self.weights[1])]
            for layer in range(1, 3):
                D[layer - 1] = (1/float(n)) * gradients[layer - 1]
                D[layer - 1][:][1:len(D[layer - 1][:])] += self.lamb * self.weights[layer - 1][:][1:len(self.weights[layer - 1][:])]
                self.weights[layer - 1] -= self.alpha * D[layer - 1]
                cost += self.lamb/float(2 * n) * np.sum(np.power(self.weights[layer - 1], 2))
            accuracy = correct / float(n)
            print self.alpha
            previousAccuracies[(iteration % len(previousAccuracies))] = accuracy
            if accuracy < (np.mean(previousAccuracies) + 0.01):
                self.alpha *= 0.75
                previousAccuracies = np.array([np.nan] * len(previousAccuracies))
            if accuracy == 1:
                return

    def classify(self, data ):
        """
        Classifies each datum as the label that most closely matches the prototype vector
        for that label.  See the project description for details.

        Recall that a datum is a util.counter...
        """
        g = np.vectorize(lambda z: 1/float(1+math.exp(-z)))
        guesses = []
        for datum in data:
            x = [None] * len(self.features)
            a = 0
            for feature in self.features:
                x[a] = datum[feature]
                a += 1
            x = np.array(x)
            a = [x]
            for layer in range(1, 3):
                a[layer - 1] = np.insert(a[layer - 1], 0, 1)
                z = np.matmul(self.weights[layer - 1], a[layer - 1])
                a.append(g(z))
            guesses.append(np.argmax(a[2]))
        return guesses


    def findHighWeightFeatures(self, label):
        """
        Returns a list of the 100 features with the greatest weight for some label
        """
        featuresWeights = []

        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()

        return featuresWeights
